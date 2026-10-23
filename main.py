
from langchain_ollama import ChatOllama

from typing import Annotated,  TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Send
from langgraph.checkpoint.memory import MemorySaver
import operator

from task_planner import create_task_plan
from blueprint_agent import (
    create_project_blueprint,
    blueprint_to_json,
)

from blueprint_agent import ProjectBlueprint
from scaffolder import scaffold_project
from worker_node import worker_node

from orchestrator import (
    initialize_task_registry,
    get_ready_tasks,
    select_next_task,
    get_execution_summary,
    update_task_status,
    record_worker_result,
)

load_dotenv()


# --------------------------------------------------
# 1. Shared workflow state
# --------------------------------------------------

class ProjectState(TypedDict):
    user_request: str
    requirements: str
    design: str
    ai_review: str
    human_decision: str
    human_feedback: str
    task_plan: str
    review_count: int
    blueprint: dict
    blueprint_json: str
    scaffold_result: dict
    task_registry: dict
    current_task: dict
    execution_summary: dict
    worker_results: Annotated[list[dict], operator.add]


# --------------------------------------------------
# 2. LLM configuration
# --------------------------------------------------
# Use one cloud model for consistent latency.
# Ensure GOOGLE_API_KEY is configured in your .env file.

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
    max_output_tokens=1800,
)

local_llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)

# --------------------------------------------------
# 3. Helpers and routing
# --------------------------------------------------

def get_text(response) -> str:
    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()

    return str(content).strip()


def route_after_review(state: ProjectState) -> str:
    # One automatic revision: review 1 -> design revision;
    # review 2 -> human approval.
    if state.get("review_count", 0) == 1:
        return "design_agent"
    if state.get("review_count", 0) >= 2:
        return "human_approval"

    raise ValueError(
        f"Unexpected review_count after review: {state.get('review_count')!r}"
    )


def route_after_human_review(state: ProjectState) -> str:
    decision = state.get("human_decision", "")

    if decision == "NEEDS_REVISION":
        return "design_agent"
    if decision == "APPROVED":
        return "task_planner"

    raise ValueError(f"Invalid human decision: {decision!r}")


# --------------------------------------------------
# 4. Requirements Agent
# --------------------------------------------------

def requirement_agent(state: ProjectState) -> dict:
    prompt = f"""
Act as a senior business analyst. Convert the user's idea into concise,
testable software requirements. Do not design architecture or write code.

USER IDEA:
{state["user_request"]}

Return these headings:
1. Functional requirements (numbered)
2. Non-functional requirements
3. User roles
4. Main use cases
5. Assumptions
6. Acceptance criteria

Be specific and concise. Avoid repeating the idea.
"""
    response = local_llm.invoke(prompt)
    requirements = get_text(response)

    if not requirements:
        raise ValueError("Requirements Agent returned empty output.")

    return {"requirements": requirements}


# --------------------------------------------------
# 5. Design Agent
# --------------------------------------------------

def design_agent(state: ProjectState) -> dict:
    review_count = state.get("review_count", 0)
    is_revision = review_count >= 1

    if is_revision:
        prompt = f"""
You are a pragmatic software architect. Revise the existing design using
the review findings and human feedback. Keep correct decisions, address
important issues, and do not add unrelated features. Do not write code.

USER IDEA:
{state["user_request"]}

REQUIREMENTS:
{state["requirements"]}

PREVIOUS DESIGN:
{state["design"]}

AI REVIEW:
{state["ai_review"]}

HUMAN FEEDBACK (may be empty):
{state.get("human_feedback", "")}

Return the complete revised design, concise and implementation-ready, with:
- Overview and chosen stack
- Architecture and component responsibilities
- Frontend
- Backend APIs
- Database entities and relationships
- Authentication, authorization, and data privacy
- Validation and error handling
- Configuration and deployment
- Testing and acceptance criteria

Only make necessary corrections. Resolve contradictions.
"""
    else:
        prompt = f"""
You are a pragmatic software architect. Produce a concise, implementation-
ready design for the user's idea. Choose a suitable, coherent technology
stack. Do not write application code or add unrelated features.

USER IDEA:
{state["user_request"]}

REQUIREMENTS:
{state["requirements"]}

Include:
- Overview and chosen stack
- Architecture and component responsibilities
- Frontend
- Backend APIs (method, route, purpose)
- Database entities, fields, and relationships
- Authentication, authorization, and data privacy
- Validation and error handling
- Configuration and deployment
- Testing and acceptance criteria

Be concrete, internally consistent, and concise.
"""
    response = local_llm.invoke(prompt)
    design = get_text(response)

    if not design:
        raise ValueError("Design Agent returned an empty design.")

    return {"design": design}


# --------------------------------------------------
# 6. AI Design Reviewer
# --------------------------------------------------

def ai_reviewer_agent(state: ProjectState) -> dict:
    current_review_count = state.get("review_count", 0)

    prompt = f"""
Act as a strict but practical software architecture reviewer.
Review the design against the user request and requirements. Do not rewrite
the design. Do not require optional enhancements unless they are important.

USER IDEA:
{state["user_request"]}

REQUIREMENTS:
{state["requirements"]}

DESIGN:
{state["design"]}

Return concise findings:
- Status: APPROVED or NEEDS_REVISION
- Missing mandatory requirements
- Blocking architecture/API/data consistency issues
- Critical security/privacy risks
- Up to 5 actionable corrections

Use NEEDS_REVISION only when material issues exist. Do not invent requirements.
"""
    response = local_llm.invoke(prompt)
    review = get_text(response)

    if not review:
        raise ValueError("AI Reviewer returned an empty review.")

    return {
        "ai_review": review,
        "review_count": current_review_count + 1,
    }


# --------------------------------------------------
# 7. Human approval interrupt
# --------------------------------------------------

def human_approval(state: ProjectState) -> dict:
    decision = interrupt({
        "message": "Review the design and its final AI review.",
        "ai_review": state["ai_review"],
        "design": state["design"],
        "options": ["APPROVED", "NEEDS_REVISION"],
    })

    return {
        "human_decision": decision["decision"],
        "human_feedback": decision.get("feedback", ""),
    }


# --------------------------------------------------
# 8. Task Planner
# --------------------------------------------------


def task_planner_node(state: ProjectState) -> dict:
    task_plan = create_task_plan(
        user_request=state["user_request"],
        requirements=state["requirements"],
        design=state["design"],
    )

    if not isinstance(task_plan, str) or not task_plan.strip():
        raise ValueError(
            "Task Planner returned an empty implementation plan."
        )

    # Define placeholders BEFORE checking the generated plan.
    placeholder_markers = [
        "[Insert product purpose]",
        "[Insert approved frontend technology]",
        "[Insert approved backend technology]",
        "[Insert approved database technology]",
        "[Insert approved testing technology]",
    ]

    found_placeholders = [
        marker
        for marker in placeholder_markers
        if marker.lower() in task_plan.lower()
    ]

    if found_placeholders:
        raise ValueError(
            "Task Planner returned a template instead of a "
            "project-specific plan. Found placeholders: "
            f"{found_placeholders}. Fix task_planner.py and retry."
        )

    print("\n========== ACTUAL TASK PLAN ==========\n")
    print(task_plan)
    print("\n========== END TASK PLAN ==========\n")

    return {"task_plan": task_plan}

# --------------------------------------------------
# 9. Blueprint Agent
# --------------------------------------------------

def blueprint_agent_node(state: ProjectState) -> dict:
    print("TASK PLAN BEING SENT TO BLUEPRINT:")
    print(state["task_plan"][:3000])
    blueprint = create_project_blueprint(
        user_request=state["user_request"],
        requirements=state["requirements"],
        design=state["design"],
        task_plan=state["task_plan"],
    )

    return {
        "blueprint": blueprint.model_dump(mode="json"),
        "blueprint_json": blueprint_to_json(blueprint),
    }

# --------------------------------------------------
# 10. Scaffolder
# --------------------------------------------------

def scaffolder_node(state: ProjectState) -> dict:
    blueprint = ProjectBlueprint.model_validate(state["blueprint"])

    result = scaffold_project(blueprint)

    if not result["success"]:
        raise RuntimeError("Project scaffolding failed.")

    return {
        "scaffold_result": result,
    }

# --------------------------------------------------
# 11. Orchestrator
# --------------------------------------------------

def orchestrator_node(state: ProjectState) -> dict:
    """
    Initialize task tracking and select the next executable task.
    Does not execute tasks or mark them as completed.
    """
    blueprint = ProjectBlueprint.model_validate(state["blueprint"])
    
    registry = state.get("task_registry")

    if not registry:
        registry = initialize_task_registry(blueprint)
    current_task = select_next_task(registry)
    summary = get_execution_summary(registry)

    print("\n========== ORCHESTRATOR ==========")
    print(f"Project: {blueprint.project_name}")
    print(f"Total tasks: {summary['total_tasks']}")
    print(f"Ready tasks: {summary['ready_task_ids']}")

    if current_task:
        print(f"Selected task: {current_task['task_id']}")
        print(f"Worker type: {current_task['worker_type']}")
        print(f"Objective: {current_task['objective']}")
    else:
        print("No task is ready for execution.")

    print("==================================\n")

    return {
        "task_registry": registry,
        "current_task": current_task or {},
        "execution_summary": summary,
    }

def dispatch_workers(state: ProjectState) -> list[Send]:
    registry = state["task_registry"]
    ready_tasks = get_ready_tasks(registry)
    dispatches = []

    for task in ready_tasks:
        update_task_status(
            registry,
            task["task_id"],
            "running",
        )

        dispatches.append(
            Send(
                "worker",
                {"task": task},
            )
        )

    return dispatches

def collect_worker_results(state: ProjectState) -> dict:
    registry = state["task_registry"]
    results = state.get("worker_results", [])

    for result in results:
        task_id = result.get("task_id")

        if task_id not in registry:
            raise KeyError(f"Unknown task ID: {task_id}")

        if registry[task_id]["status"] != "running":
            continue

        record_worker_result(registry, result)

    return {
        "task_registry": registry,
        "execution_summary": get_execution_summary(registry),
    }

def route_after_collection(state: ProjectState) -> str:
    summary = get_execution_summary(state["task_registry"])

    if summary["all_completed"] or summary["has_failures"]:
        return "end"

    if summary["ready_task_ids"]:
        return "orchestrator"

    return "end"
# --------------------------------------------------
# 11. Build graph
# --------------------------------------------------

builder = StateGraph(ProjectState)

builder.add_node("requirement_agent", requirement_agent)
builder.add_node("design_agent", design_agent)
builder.add_node("ai_reviewer_agent", ai_reviewer_agent)
builder.add_node("human_approval", human_approval)
builder.add_node("task_planner", task_planner_node)
builder.add_node("blueprint_agent", blueprint_agent_node)
builder.add_node("scaffolder", scaffolder_node)
builder.add_node("orchestrator", orchestrator_node)
builder.add_node("worker", worker_node)
builder.add_node("collect_worker_results", collect_worker_results)

builder.add_edge(START, "requirement_agent")
builder.add_edge("requirement_agent", "design_agent")
builder.add_edge("design_agent", "ai_reviewer_agent")

builder.add_conditional_edges(
    "ai_reviewer_agent",
    route_after_review,
    {
        "design_agent": "design_agent",
        "human_approval": "human_approval",
    },
)

builder.add_conditional_edges(
    "human_approval",
    route_after_human_review,
    {
        "design_agent": "design_agent",
        "task_planner": "task_planner",
    },
)

builder.add_edge("task_planner", "blueprint_agent")
builder.add_edge("blueprint_agent", "scaffolder")
builder.add_edge("scaffolder", "orchestrator")

builder.add_conditional_edges(
    "orchestrator",
    dispatch_workers,
    ["worker"],
)

builder.add_edge("worker", "collect_worker_results")

builder.add_conditional_edges(
    "collect_worker_results",
    route_after_collection,
    {
        "orchestrator": "orchestrator",
        "end": END,
    },
)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
