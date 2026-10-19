from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

load_dotenv()


# --------------------------------------------------
# LLM configuration
# --------------------------------------------------

# planner_llm = ChatOllama(
#     model="llama3.2",
#     temperature=0,
# )

planner_llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    num_predict=1200,
    repeat_penalty=1.1,
)

# planner_llm = ChatGroq(
#     model="openai/gpt-oss-120b",
#     temperature=0,
# )

# --------------------------------------------------
# Task Planner
# --------------------------------------------------

def create_task_plan(
    user_request: str,
    requirements: str,
    design: str,
) -> str:
    """
    Generate an agent-oriented implementation plan from the approved design.

    The plan is intended for downstream Blueprint, Scaffolder, and coding
    agents. It is not a human project schedule and contains no implementation
    code.
    """

  
    prompt = f"""
    You are the Task Planning Agent in an AI software development system.

    Your output is consumed by automated Blueprint, Scaffolder, and coding agents.
    Produce a precise, machine-actionable implementation plan based ONLY on the
    user request, requirements, and approved technical design.

    ==================== SOURCE INPUTS ====================

    USER REQUEST:
    {user_request}

    REQUIREMENTS:
    {requirements}

    APPROVED TECHNICAL DESIGN:
    {design}

    ==================== CORE OBJECTIVE ====================

    Create a concise, dependency-aware implementation plan for a small
    application. Prefer 6-10 meaningful tasks when appropriate.

    The approved technical design is authoritative for the technology stack,
    architecture, database, and APIs. Requirements define required behavior.

    Do not silently replace approved technologies or add unsupported services,
    frameworks, features, database systems, or deployment platforms.

    If the design contains contradictions or unresolved decisions, record them
    under assumptions or unresolved decisions instead of inventing a solution.

    Do not write implementation code, project schedules, dates, sprint durations,
    staffing estimates, or generic project-management advice.

    ==================== REQUIRED OUTPUT STRUCTURE ====================

    Return Markdown with exactly these top-level sections.

    # Implementation Plan

    ## 1. Project Summary

    Include:
    - Product purpose
    - Approved frontend technology
    - Approved backend technology
    - Approved database technology
    - Approved testing technology
    - Architecture style, or "Not specified"
    - Explicit assumptions and unresolved decisions

    Do not change the approved technology stack.

    ## 2. Project Structure

    Propose a repository-relative directory tree consistent with the approved
    design. Include appropriate frontend, backend, database, tests, and
    configuration directories only when relevant.

    Use concrete proposed file paths consistently throughout this plan.

    The directory tree is a proposal, not a list of implementation tasks.
    Do not put task IDs in this section.

    ## 3. Database Tasks

    Create individual database task blocks using IDs DB-001, DB-002, etc.

    Every task MUST use this exact format:

    ### DB-001: Short task title
    - Task ID: DB-001
    - Objective: One specific, independently verifiable objective.
    - Files to create or modify:
    - path/to/file
    - Entities, fields, relationships, indexes, and constraints:
    - Specify only what is supported by the inputs.
    - Dependencies (task IDs): None
    - Acceptance criteria:
    - One or more concrete, verifiable checks.

    Repeat the complete block for each database task.

    Each task ID MUST appear in its own block. NEVER group IDs such as
    "DB-001, DB-002" in one Task ID field.

    Define database models and schema before tasks that depend on them.
    Use only the database technology specified by the approved design.
    Do not introduce SQL tables into a document-database design or collections
    into a relational-database design unless the approved design requires them.

    ## 4. Backend Tasks

    Create individual backend task blocks using IDs BE-001, BE-002, etc.

    Every task MUST use this exact format:

    ### BE-001: Short task title
    - Task ID: BE-001
    - Objective: One specific, independently verifiable objective.
    - Files to create or modify:
    - path/to/file
    - API contract:
    - HTTP method and route
    - Purpose
    - Request fields
    - Response shape and status codes when supported by the design
    - Authentication, authorization, validation, and error handling:
    - Specify relevant requirements.
    - Dependencies (task IDs): DB-001
    - Acceptance criteria:
    - One or more concrete, verifiable checks.

    Repeat the complete block for each backend task.

    Every backend task must have its own unique ID and objective.
    Do not group multiple IDs into one task.

    Keep endpoint names, entity names, fields, and response contracts consistent
    with the database and approved design.

    Do not create unnecessary APIs or invent unspecified business rules.

    ## 5. Frontend Tasks

    Create individual frontend task blocks using IDs FE-001, FE-002, etc.

    Every task MUST use this exact format:

    ### FE-001: Short task title
    - Task ID: FE-001
    - Objective: One specific, independently verifiable objective.
    - Files to create or modify:
    - path/to/file
    - Pages, components, forms, navigation, and state:
    - Specify the relevant UI behavior.
    - Backend endpoints consumed:
    - List only endpoints defined in the backend tasks.
    - Dependencies (task IDs): BE-001
    - Acceptance criteria:
    - One or more concrete, verifiable checks.

    Repeat the complete block for each frontend task.

    Do not group multiple IDs into one task. Follow the approved frontend stack.
    Do not add frameworks or libraries that are not supported by the design.

    Frontend tasks must depend on backend tasks only when they genuinely require
    the backend contract or implementation to be available.

    ## 6. Integration and Testing Tasks

    Create individual integration/testing task blocks using IDs INT-001,
    INT-002, etc. Use TEST-001, TEST-002, etc. only if a distinct testing task
    category is necessary. Prefer INT IDs for integration and testing tasks
    in this plan.

    Every task MUST use this exact format:

    ### INT-001: Short task title
    - Task ID: INT-001
    - Objective: One specific testing or integration objective.
    - Files to create or modify:
    - path/to/test/file
    - Test scope:
    - Unit, API, integration, authorization, data isolation, security,
        or configuration checks relevant to the approved design.
    - Dependencies (task IDs): BE-001
    - Acceptance criteria:
    - One or more concrete, verifiable checks.

    Repeat the complete block for every integration/testing task.

    CRITICAL:
    - Section 6 MUST contain explicit task blocks, not just a list of test files.
    - Every INT task must have exactly one ID and a specific objective.
    - Every INT task must list its files, dependencies, and acceptance criteria.
    - Do not invent TEST IDs if INT IDs are used.
    - Do not create testing tasks for unsupported features or services.

    ==================== TASK ID AND DEPENDENCY CONTRACT ====================

    These rules are mandatory because downstream agents parse this output.

    1. Every task is a separate, complete task block with exactly one Task ID.
    2. Use only these ID prefixes:
    DB-001, DB-002, ...
    BE-001, BE-002, ...
    FE-001, FE-002, ...
    INT-001, INT-002, ...
    3. Number tasks sequentially within each category, starting at 001.
    4. Never group multiple task IDs on one line or in one task block.
    5. Never omit a task's objective, file list, dependencies, or acceptance
    criteria.
    6. Every task ID must be unique across the entire plan.
    7. Every dependency must reference a task ID declared in one of the task
    blocks in this plan.
    8. Use "None" when a task has no dependencies.
    9. Do not reference task IDs that have not been defined.
    10. Do not create circular dependencies. If task A depends on task B,
        task B must not depend on task A, directly or indirectly.
    11. Database tasks must not depend on backend tasks when the backend depends
        on those database models or schemas.
    12. Frontend tasks must not depend on backend tasks that themselves depend
        on those frontend tasks.
    13. Integration tests may depend on the components they test, but the
        components must not depend on those integration tests.
    14. Do not infer dependencies merely from task order. Declare only genuine
        prerequisites.
    15. Use the exact same file path spelling and casing wherever a path appears.
    16. Every task must reference at least one proposed repository-relative file.
    17. Keep each task small enough to implement and test independently.
    18. Do not silently omit required functionality because the plan is concise.

    Before returning the plan, internally verify:
    - Every task ID has exactly one complete task block.
    - No IDs are grouped in a single Task ID field.
    - All dependencies reference existing tasks.
    - No dependency cycles exist.
    - All task file paths are repository-relative and consistent.
    - All required task fields are present.
    - All task IDs mentioned in Execution Order are defined.
    - All task IDs mentioned in Handoff are defined.
    - The integration/testing section contains explicit task blocks.
    - No unsupported technology or feature was introduced.

    ==================== SECTION 7: EXECUTION ORDER ====================

    ## 7. Execution Order

    List every declared task ID exactly once, in dependency-respecting order.

    Use this format:

    1. DB-001
    2. DB-002
    3. BE-001
    4. FE-001
    5. INT-001

    Use the actual IDs generated in the task blocks; the example above is
    illustrative only.

    Tasks may run in parallel only when their dependencies permit it.
    Do not create new task IDs in this section.
    Do not omit any declared task.

    ==================== SECTION 8: HANDOFF ====================

    ## 8. Handoff to the Project Blueprint Generator

    Summarize:
    - Approved technology decisions
    - Required components
    - Required entities and relationships
    - API contracts
    - Required environment variable NAMES only; never include secrets
    - Repository structure
    - Task IDs and their declared dependencies
    - Assumptions requiring resolution

    Use task IDs exactly as declared in the task blocks.
    Do not create, rename, group, or omit task IDs in the handoff.

    ==================== FINAL RULES ====================

    1. Treat the approved technical design as authoritative.
    2. Treat user requirements as authoritative for required functionality.
    3. Do not replace approved technologies with personal preferences.
    4. Do not invent business rules, fields, APIs, credentials, or services.
    5. Mark uncertain file paths as proposed.
    6. Keep entity names, fields, API routes, and response contracts consistent.
    7. Use environment variable names only; never output secret values.
    8. Return only the structured Implementation Plan.
    9. Do not include source-code implementations.
    10. Do not group task IDs under a shared task block.
    11. The Task ID field must contain exactly one ID.
    12. Every implementation task must have its own explicit task block.
    """


    response = planner_llm.invoke(prompt)

    if isinstance(response.content, str):
        task_plan = response.content.strip()
    elif isinstance(response.content, list):
        task_plan = "\n".join(
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    else:
        task_plan = str(response.content).strip()

    if not task_plan:
        raise ValueError("Task Planner returned an empty implementation plan.")

    return task_plan
