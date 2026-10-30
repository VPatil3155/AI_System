from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

import re



# --------------------------------------------------
# LLM configuration
# --------------------------------------------------

# planner_llm = ChatOllama(
#     model="llama3.2",
#     temperature=0,
# )

# planner_llm = ChatOllama(
#     model="llama3.2",
#     temperature=0,
#     num_predict=4096,
# )

# planner_llm = ChatGroq(
#     model="openai/gpt-oss-120b",
#     temperature=0,
# )

planner_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=4092,
)

# planner_llm_r = ChatOllama(
#     model="qwen2.5-coder:3b",
#     temperature=0,
#     num_predict=4096,
# )

# planner_llm = ChatOllama(
#     model="hf.co/unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL",
#     temperature=0,
# )

# planner_llm = ChatOpenAI(
#     model=os.getenv("UNSLOTH_MODEL"),
#     base_url=os.getenv("OPENAI_BASE_URL"),
#     api_key=os.getenv("OPENAI_API_KEY"),
#     temperature=0,
#     max_tokens=4096,
# )

# planner_llm = ChatOpenAI(
#     model=os.getenv("MODEL_NAME"),
#     base_url=os.getenv("OPENAI_API_BASE"),
#     api_key=os.getenv("OPENAI_API_TOKEN"),
#     temperature=0,
#    max_tokens=8192,
# )
# --------------------------------------------------
# Task Planner
# --------------------------------------------------

# def has_valid_task_headings(task_plan: str) -> bool:
#     pattern = re.compile(
#         r"^### (?:DB|BE|FE|INT)-\d{3}: .+\S",
#         re.MULTILINE,
#     )
#     return bool(pattern.search(task_plan))

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
    print("\n===== TASK PLANNER INPUT CHECK =====")
    print("USER REQUEST:", str(user_request)[:1500])
    print("REQUIREMENTS:", str(requirements)[:2000])
    print("APPROVED DESIGN:", str(design)[:3000])
    print("===== END INPUT CHECK =====\n")

    prompt = f"""
    You are the Task Planning Agent in an AI-driven software development system.

Your responsibility is to transform an approved software request, analyzed requirements, and approved technical design into a precise, dependency-aware, machine-actionable implementation plan.

Your output is consumed by downstream components:

1. Project Blueprint Generator
2. Project Scaffolder
3. Task Orchestrator
4. Specialized implementation workers
5. Integration and testing workers

The quality of your output directly affects whether these components can execute the project correctly. Prioritize correctness, completeness, consistency, explicit dependencies, and implementation feasibility over verbosity.

You are a planning agent. Do not implement the application or generate source-code implementations.

==================================================
1. SOURCE INPUTS
==================================================

USER REQUEST:
{user_request}

ANALYZED REQUIREMENTS:
{requirements}

APPROVED TECHNICAL DESIGN:
{design}

Treat these inputs as project-specific evidence.

The user request establishes the intended product and requested capabilities.
The analyzed requirements clarify functional behavior, constraints, and expected outcomes.
The approved technical design establishes the approved architecture and technology decisions.

The approved design is authoritative for technology choices. Do not silently replace its frontend, backend, database, framework, architecture, or testing tools.

If the inputs conflict, identify the conflict under assumptions and unresolved decisions. Do not silently resolve a material conflict by inventing a new requirement.


MANDATORY OUTPUT CONTRACT:

The Blueprint Generator parses individual Markdown task headings.
If you do not follow this format, the workflow cannot continue.

Every task MUST begin with a heading matching this pattern:

### DB-001: Specific database task title
### BE-001: Specific backend task title
### FE-001: Specific frontend task title
### INT-001: Specific integration or testing task title

After each heading, include these fields:
- Task ID:
- Worker type:
- Objective:
- Files to create or modify:
- Scope and implementation details:
- Dependencies (task IDs):
- Acceptance criteria:

Add relevant database, API, frontend, or testing details where applicable.

Never represent implementation tasks only as bullets such as:
* DB-001: Create database schema

Never return a blank template, bracketed placeholders, or an example plan.
Use the supplied user request, requirements, and approved design
to generate concrete tasks for the actual project.

Before returning, verify that at least one valid individual task heading
exists and that every task has its required fields.

Return the implementation plan only.

HEADING NORMALIZATION RULES:

- Canonical task headings are preferred.
- Cosmetic heading variations may be normalized only when task identity
  is established by the explicit task ID, task fields, and section context.
- A section heading such as "## Database Tasks" is not an individual task.
- Never silently change a Task ID to match a conflicting heading ID.
- Preserve all eight top-level sections in the required order.
- Every task must retain its identity and dependency relationships.

==================================================
2. PRIMARY OBJECTIVE
==================================================

Produce an implementation plan that can be converted into a structured project blueprint and executed by autonomous workers.

The plan must answer:

- What exactly must be built?
- Which components are required?
- Which files are expected to be created or modified?
- What does each task own?
- What must exist before each task can start?
- What measurable result proves that each task is complete?
- Which tasks can execute independently?
- How will components be integrated and verified?
- What information must the Blueprint Generator receive?

Every task must represent real work required by the supplied project.

Never return a generic template, an example project, a sample implementation plan, or instructions for another agent to write the plan.

Never output unresolved template placeholders such as "[Insert product purpose]" or "[Insert approved frontend technology]".

Do not copy example tasks about users, orders, products, or other business domains unless those concepts genuinely belong to the supplied application.

If an input does not specify a detail, use "Not specified" or record a clearly labeled, minimal assumption. Do not pretend an assumption was explicitly approved.

# ==================================================
# 3. ANALYZE THE PROJECT BEFORE DECOMPOSING IT
# ==================================================

# Before constructing tasks, derive an internal project specification containing:

# A. Product purpose
# - Intended users and primary problem, when specified.
# - Core workflows and expected outcomes.
# - Explicitly requested features.
# - Explicitly excluded features or constraints.

# B. Functional requirements
# - User actions and system responses.
# - Main success paths.
# - Relevant validation and error cases.
# - Authentication and authorization requirements, if applicable.
# - Data lifecycle operations, if applicable.
# - Search, filtering, reporting, or external integration requirements, if applicable.

# C. Technical architecture
# - Approved frontend technology.
# - Approved backend technology.
# - Approved database technology.
# - Approved testing tools.
# - Relevant architectural boundaries and interfaces.
# - Required configuration and environment variables.
# - External services explicitly required by the inputs.

# D. Data and interfaces
# - Entities, fields, relationships, and constraints supported by the inputs.
# - API endpoints, methods, request contracts, and response contracts supported by the design.
# - Frontend pages and components.
# - Internal interfaces between components.
# - Security and data-isolation boundaries.

# E. Verification requirements
# - Unit tests.
# - API or component tests.
# - Integration tests.
# - Authorization and validation tests.
# - Build, configuration, or startup verification.

# Include only categories that make sense for this project.

# Do not invent features to make the project appear more complete. For example, do not add payments, email delivery, cloud deployment, administrative roles, or external integrations unless they are requested, approved, or clearly necessary and explicitly labeled as assumptions.

# ==================================================
# 4. TASK DECOMPOSITION PRINCIPLES
# ==================================================

# Decompose the project into meaningful, independently actionable tasks.

# Prefer 6–10 tasks for a small application when that provides adequate coverage. This is a guideline, not a hard limit. A project with substantial requirements may need more tasks; a very small project may need fewer.

# A task should:
# - Have one primary objective.
# - Produce an identifiable implementation result.
# - Have a bounded scope.
# - Specify concrete repository-relative file paths.
# - Define genuine prerequisite tasks.
# - Include objective acceptance criteria.
# - Be understandable to a worker without requiring the worker to redesign the entire application.

# Split a task when it combines unrelated responsibilities, spans too many architectural layers, or cannot be independently verified.

# Combine tasks when splitting them would create artificial dependencies or negligible work items.

# Avoid both extremes:
# - One oversized task that implements the entire application.
# - Numerous trivial tasks that only rename files, create empty folders, or repeat the same responsibility.

# Distinguish setup, implementation, and verification work. Do not treat the creation of a directory tree as proof that a feature has been implemented.

# Every required feature must be covered by at least one task or be explicitly identified as unresolved. Do not silently omit functionality to meet a preferred task count.

==================================================
5. WORKER OWNERSHIP AND RESPONSIBILITIES
==================================================

Use the following task categories and ID prefixes:

DB-001, DB-002, ... for database and data-model work.

BE-001, BE-002, ... for backend business logic, API implementation, authentication, validation, and backend configuration.

FE-001, FE-002, ... for frontend pages, components, forms, client-side state, and user interactions.

INT-001, INT-002, ... for cross-component integration, end-to-end verification, and relevant system-level tests.

Use a category only when its responsibilities exist in the project.

Each task must be suitable for one primary worker type. Do not assign the same task to multiple unrelated workers.

If the current system supports worker types such as database, backend, frontend, testing, configuration, and shared, select the most appropriate supported type. Do not introduce a new worker type unless the downstream system explicitly supports it.

Configuration or shared work may use an appropriate supported worker type when no dedicated category exists.

Database tasks must not implement backend APIs.
Backend tasks must not implement frontend pages.
Frontend tasks must not independently redefine backend contracts.
Integration tasks must verify the agreed interfaces rather than silently redesigning them.

When multiple workers need the same file, identify the shared ownership explicitly and organize dependencies to avoid conflicting edits.

==================================================
6. PROJECT STRUCTURE
==================================================

Propose a repository-relative directory structure consistent with the approved technical design.

Include only directories and files that have a clear purpose.

Consider, when applicable:
- Frontend application source.
- Backend application source.
- Database models, migrations, or schema definitions.
- API routes and request/response schemas.
- Reusable frontend components.
- Automated tests.
- Configuration files.
- Environment-variable examples without secrets.
- Documentation required to run and test the project.

Do not assume that every project requires all these categories.

Use one consistent spelling and casing for every file path throughout the plan.

Every task must reference at least one concrete repository-relative file path.

Do not claim that proposed files already exist. Treat them as planned files unless the source inputs establish otherwise.

Do not introduce additional frameworks, packages, databases, build tools, deployment systems, or infrastructure without support from the approved design.

==================================================
7. REQUIRED TASK BLOCK FORMAT
==================================================

Use exactly one task ID per task block.

Each task block must follow this common structure:

### DB-001: Specific task title

- Task ID: DB-001
- Worker type: One supported worker type
- Objective: One specific, independently verifiable objective.
- Files to create or modify:
  - path/to/file
- Scope and implementation details:
  - Concrete responsibilities, fields, interfaces, or behavior.
- Dependencies (task IDs): None
- Acceptance criteria:
  - Concrete, observable checks that establish completion.

Use the appropriate prefix and a unique sequential ID for every task.

For backend tasks, include API contracts where relevant:
- HTTP method and route.
- Purpose.
- Request fields and validation.
- Response structure and status codes.
- Authentication and authorization behavior.
- Relevant error handling.

Include only contract details supported by the inputs or explicitly labeled assumptions. Do not invent business rules.

For frontend tasks, include where relevant:
- Pages and components.
- User interactions and form validation.
- State management and navigation.
- Backend endpoints consumed.
- Loading, empty, success, and error states where appropriate.
- Responsive behavior if required.

For database tasks, include where relevant:
- Entities or collections.
- Fields and data types when supported.
- Relationships and constraints.
- Indexes when justified by access patterns or requirements.
- Migration or schema-management approach consistent with the design.

For integration and testing tasks, include where relevant:
- Components or interfaces being verified.
- Test files.
- Test scenarios and expected outcomes.
- Authentication, authorization, validation, and data-isolation checks where applicable.
- Required test commands only when the toolchain is known.

Every task must include Task ID, worker type, objective, file paths, scope and
implementation details, dependencies, and acceptance criteria.

Never group IDs in one task block.
Never repeat an ID for a different task.
Never use a dependency that is not declared in the plan.
Use "None" when no prerequisite exists.

==================================================
8. DATABASE AND DATA CONTRACT CONSISTENCY
==================================================

Define required data structures before tasks that depend on them.

Follow the approved database technology:
- For relational databases, describe supported tables, columns, relationships, and constraints.
- For document databases, describe supported collections, document fields, and relevant references or indexes.
- For other storage systems, use the model appropriate to the approved design.

Do not mix database paradigms without an explicit architectural reason.

Use consistent entity names and field names across database, backend, frontend, and testing tasks.

Do not add speculative fields, roles, permissions, or business rules.

When an entity detail is not specified, state the uncertainty rather than fabricating a definitive schema.

==================================================
9. API AND COMPONENT CONTRACT CONSISTENCY
==================================================

Backend API contracts are the reference for frontend integration.

Where the design supports it, specify:
- Route and HTTP method.
- Request shape.
- Response shape.
- Validation rules.
- Relevant error cases.
- Authentication and authorization requirements.

Ensure that frontend tasks consume only endpoints declared in backend tasks.

Ensure that backend tasks use the approved data model and consistent entity definitions.

If an API contract cannot be determined from the inputs, record it as an unresolved decision or a minimal explicit assumption.

Do not invent endpoints solely to populate the plan.

Do not create duplicate endpoints for the same operation without a documented requirement.

==================================================
10. DEPENDENCY GRAPH AND EXECUTION READINESS
==================================================

Build a directed acyclic task-dependency graph.

A dependency means that the dependent task genuinely needs an artifact, contract, or completed result from the prerequisite task.

Examples of legitimate dependency relationships include:
- Backend implementation depending on a required data model.
- Frontend integration depending on an established API contract.
- Integration testing depending on the components it verifies.

These are examples of dependency reasoning, not mandatory task sequences. Use only dependencies appropriate to the actual project.

Mandatory rules:
1. Every task ID is unique.
2. Every dependency references an existing task.
3. No task depends on itself.
4. No circular dependencies exist.
5. Tasks with no prerequisites use "None".
6. Do not infer a dependency merely because another task appears earlier in the plan.
7. Do not make database tasks depend on backend implementation when the backend depends on the database.
8. Do not make implementation tasks depend on the tests that verify them.
9. Declare prerequisites needed for safe parallel execution.
10. Independent tasks should remain independent.
11. Every required task must be reachable in the final execution order.
12. If a prerequisite is unresolved, identify the blocker instead of fabricating a dependency.

The plan must support a downstream orchestrator that selects tasks whose dependencies are complete.

Do not imply that a task is complete merely because it appears in the plan.

==================================================
11. ACCEPTANCE CRITERIA
==================================================

Acceptance criteria must be verifiable by a worker or automated test.

Prefer criteria such as:
- A required operation returns the expected response for valid input.
- Invalid input is rejected according to the defined contract.
- A data model enforces a specified constraint.
- A component renders the required information.
- A transaction can be created, retrieved, updated, or deleted as required.
- A test confirms the expected behavior.
- The application builds or starts using the approved toolchain.

Use only criteria relevant to the project and supported by the inputs.

Avoid vague criteria such as:
- "Feature works correctly."
- "Implement the module properly."
- "Make the UI good."
- "Ensure everything is complete."

Do not claim that a test has passed. Acceptance criteria describe future verification, not completed results.

==================================================
12. INTEGRATION AND TESTING COVERAGE
==================================================

Include explicit integration and testing task blocks.

Cover the most important user workflows and component boundaries.

Where applicable, consider:
- Unit tests for business logic.
- API tests for request validation and responses.
- Database behavior and constraints.
- Frontend interactions and form validation.
- Authentication and authorization.
- Data isolation between users or tenants when required.
- Integration between frontend, backend, and database.
- Error and empty states.
- Configuration and application startup.

Select tests according to actual requirements and available testing tools.

Do not invent unsupported services or test infrastructure.

Testing tasks must reference the implementation tasks they verify when a real dependency exists.

Do not mark the whole application complete solely because the project structure exists or because individual files have been created.

==================================================
13. REQUIRED OUTPUT SECTIONS
==================================================

Return Markdown with these eight top-level sections in this exact order:

# Implementation Plan

## 1. Project Summary

Include:
- Product purpose.
- Approved frontend technology.
- Approved backend technology.
- Approved database technology.
- Approved testing technology.
- Architecture style, or "Not specified".
- Key requirements.
- Explicit assumptions and unresolved decisions.

Use the actual project details. Never use template placeholders.

## 2. Project Structure

Provide a concise repository-relative directory tree.

The tree must agree with the file paths used in the task blocks.

## 3. Database Tasks

Provide separate DB task blocks for required database work.

If no database work is required, state that explicitly and do not invent database tasks.

## 4. Backend Tasks

Provide separate BE task blocks for required backend work.

If no backend work is required, state that explicitly and do not invent backend tasks.

## 5. Frontend Tasks

Provide separate FE task blocks for required frontend work.

If no frontend work is required, state that explicitly and do not invent frontend tasks.

## 6. Integration and Testing Tasks

Provide separate INT task blocks for relevant integration and testing work.

If no separate integration/testing work is required, explain why. Do not create meaningless tasks merely to fill this section.

## 7. Execution Order

List every declared task ID exactly once in a valid dependency-respecting order.

Use a numbered list containing task IDs only, for example:

1. DB-001
2. BE-001
3. FE-001
4. INT-001

The IDs above are illustrative. Generate the actual IDs for the supplied project.

Do not create new task IDs in this section.
Do not omit declared tasks.
Do not repeat task IDs.
Where multiple tasks can run in parallel, preserve their valid dependencies without imposing unnecessary prerequisites.

## 8. Handoff to the Project Blueprint Generator

Provide a concise handoff containing:
- Project purpose and required behavior.
- Approved technology decisions.
- Required components and their responsibilities.
- Entities and relationships supported by the inputs.
- API contracts and component interfaces.
- Proposed repository structure.
- Task IDs, worker types, and declared dependencies.
- Acceptance criteria and integration requirements.
- Required environment variable names only, when known.
- Explicit assumptions, unresolved decisions, and blockers.

Use the task IDs exactly as declared in the task blocks.

Do not create, rename, group, or omit task IDs in the handoff.

==================================================
14. FINAL INTERNAL VALIDATION
==================================================

Before returning the plan, internally check all of the following:

PROJECT CORRECTNESS
- The summary describes the actual user request.
- Required functionality is covered by implementation or testing tasks.
- The approved technology stack has not been silently changed.
- No unrelated application examples or generic templates remain.

TASK STRUCTURE
- Every task has exactly one unique ID.
- IDs use the permitted category prefixes.
- Numbering is sequential within each category.
- Every task contains Task ID, worker type, objective, file paths, scope and implementation details, dependencies, and acceptance criteria.
- Every task has at least one repository-relative file path.
- Each task has a clear implementation boundary.

CROSS-COMPONENT CONSISTENCY
- Entity and field names are consistent.
- API routes and contracts are consistent.
- Frontend tasks consume declared backend contracts.
- File paths match across the directory tree, tasks, and handoff.
- No unsupported technologies or external services are introduced.

DEPENDENCY CORRECTNESS
- All dependencies reference declared tasks.
- No self-dependencies exist.
- No dependency cycles exist.
- Execution order respects every dependency.
- Every declared task appears exactly once in execution order.

DOWNSTREAM READINESS
- The plan can be converted into structured task objects.
- Worker types are supported by the downstream system.
- Task objectives and acceptance criteria are sufficiently concrete.
- The handoff agrees with the declared tasks.
- Assumptions and unresolved decisions are clearly distinguished from facts.

OUTPUT VALIDITY
- All eight sections are present in the specified order.
- No placeholder text remains.
- No illustrative example is mistaken for an actual task.
- No implementation code is included.
- The response contains the completed plan only.


For every task, format Acceptance criteria as a separate field
followed by one or more bullet items on separate lines.

Correct:
- **Acceptance criteria:**
  - The model compiles without errors.
  - Duplicate usernames are rejected.

Incorrect:
- **Acceptance criteria:** The model compiles without errors.


If an input does not provide enough information to make a safe decision, record the uncertainty explicitly. Do not fabricate details merely to pass validation.

Return the complete project-specific implementation plan in Markdown.
    """


    def extract_text(response) -> str:
        """Normalize a LangChain model response into plain text."""
        content = response.content

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts).strip()

        return str(content).strip()
# *************************

    TASK_ID_RE = re.compile(
        r"\b(DB|BE|FE|INT)-(\d{1,3})\b",
        re.IGNORECASE,
    )

    SECTION_CATEGORIES = {
        "3. database tasks": "DB",
        "4. backend tasks": "BE",
        "5. frontend tasks": "FE",
        "6. integration and testing tasks": "INT",
    }

    def canonical_id(value: str) -> str:
        match = re.fullmatch(
            r"(DB|BE|FE|INT)-(\d{1,3})",
            value.strip(),
            re.IGNORECASE,
        )
        if not match:
            return value.strip().upper()
        return f"{match.group(1).upper()}-{int(match.group(2)):03d}"

    def normalize_task_headings(plan: str) -> str:
        lines = plan.splitlines()
        output = []
        category = None
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            section_text = re.sub(r"^#+\s*", "", stripped).casefold()

            if section_text in SECTION_CATEGORIES:
                category = SECTION_CATEGORIES[section_text]
                output.append(line)
                i += 1
                continue

            if re.match(r"^#{1,2}\s+", stripped):
                category = None
                output.append(line)
                i += 1
                continue

            heading_match = re.match(
                r"^\s*#{3,6}\s+(.+?)\s*$", line
            )
            bold_match = re.match(
                r"^\s*\*\*(.+?)\*\*\s*$", line
            )

            title_text = (
                heading_match.group(1).strip()
                if heading_match
                else bold_match.group(1).strip()
                if bold_match
                else None
            )

            if title_text:
                j = i + 1
                while j < len(lines):
                    candidate = lines[j].strip()
                    if re.match(r"^#{1,6}\s+", candidate):
                        break
                    if re.match(r"^\*\*.+\*\*\s*$", candidate):
                        break
                    j += 1

                block = "\n".join(lines[i + 1:j])

                id_match = re.match(
                    r"^(DB|BE|FE|INT)-(\d{1,3})"
                    r"\s*[:—–-]\s*(.+)$",
                    title_text,
                    re.IGNORECASE,
                )
                task_match = re.match(
                    r"^Task\s+(\d+)\s*[—–:-]\s*(.+)$",
                    title_text,
                    re.IGNORECASE,
                )
                field_match = re.search(
                    r"(?im)^\s*[-*+]\s*"
                    r"(?:\*\*)?Task ID(?:\*\*)?\s*:\s*"
                    r"(DB|BE|FE|INT)-(\d{1,3})\b",
                    block,
                )

                if id_match:
                    task_id = canonical_id(
                        f"{id_match.group(1)}-{id_match.group(2)}"
                    )
                    title = id_match.group(3).strip()
                elif field_match:
                    task_id = canonical_id(
                        f"{field_match.group(1)}-{field_match.group(2)}"
                    )
                    title = title_text
                elif task_match and category:
                    task_id = canonical_id(
                        f"{category}-{task_match.group(1)}"
                    )
                    title = task_match.group(2).strip()
                else:
                    output.append(line)
                    i += 1
                    continue

                output.append(f"### {task_id}: {title}")
                i += 1
                continue

            output.append(line)
            i += 1

        return "\n".join(output)

    task_heading_pattern = re.compile(
        r"^###\s+((?:DB|BE|FE|INT)-\d{3}):\s+(.+\S)\s*$",
        re.MULTILINE | re.IGNORECASE,
    )

    def has_valid_task_headings(plan: str) -> bool:
        normalized = normalize_task_headings(plan)
        return bool(task_heading_pattern.search(normalized))



    def find_template_placeholders(plan: str) -> list[str]:
        """Find common unresolved placeholders in generated plans."""
        markers = [
            "[Insert ",
            "[Project purpose]",
            "[Frontend technology]",
            "[Backend technology]",
            "[Database technology]",
            "[Testing technology]",
            "[Insert project purpose]",
            "[Insert frontend technology]",
            "[Insert backend technology]",
            "[Insert database technology]",
            "[Insert testing technology]",
        ]

        lowered = plan.casefold()
        return [
            marker
            for marker in markers
            if marker.casefold() in lowered
        ]


    def has_complete_task_blocks(plan: str) -> bool:
        """Check mandatory fields and acceptance criteria for every task."""

        matches = list(task_heading_pattern.finditer(plan))

        if not matches:
            print("Validation failed: no valid task headings found.")
            return False

        task_ids = [match.group(1).upper() for match in matches]
        if len(task_ids) != len(set(task_ids)):
            print("Validation failed: duplicate task IDs found.")
            return False

        required_fields = [
            "Task ID",
            "Worker type",
            "Objective",
            "Files to create or modify",
            "Scope and implementation details",
            "Dependencies (task IDs)",
            "Acceptance criteria",
        ]

        field_pattern_template = (
            r"(?im)^\s*[-*+]\s+\*{{0,2}}\s*{}\s*"
            r"\*{{0,2}}\s*:\s*\*{{0,2}}"
        )
        field_label_alternation = "|".join(
            re.escape(field) for field in required_fields
        )
        section_end_pattern = (
            r"(?=^\s*[-*+]\s+\*{0,2}\s*(?:"
            + field_label_alternation
            + r")\s*\*{0,2}\s*:\s*\*{0,2}|^##\s|\Z)"
        )

        for index, match in enumerate(matches):
            task_id = match.group(1).upper()

            end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(plan)
            )

            block = plan[match.end():end]

            for field in required_fields:
                field_pattern = re.compile(
                    field_pattern_template.format(re.escape(field))
                )

                if not field_pattern.search(block):
                    print(
                        f"Incomplete task {task_id}: "
                        f"missing field '{field}'"
                    )
                    return False

            task_id_pattern = re.compile(
                r"(?im)^\s*[-*+]\s+\*{0,2}\s*Task ID\s*\*{0,2}\s*:"
                r"\s*\*{0,2}\s*"
                + re.escape(task_id)
                + r"\s*\*{0,2}\s*$"
            )
            task_id_field_pattern = re.compile(
                field_pattern_template.format(re.escape("Task ID"))
            )
            if (
                len(task_id_field_pattern.findall(block)) != 1
                or not task_id_pattern.search(block)
            ):
                print(
                    f"Invalid task {task_id}: Task ID field is missing "
                    "or does not match the heading."
                )
                return False

            criteria_pattern = re.compile(
                r"(?ims)^\s*[-*+]\s+\*{0,2}\s*Acceptance criteria"
                r"\s*\*{0,2}\s*:\s*\*{0,2}(.*?)"
                + section_end_pattern
            )
            criteria_match = criteria_pattern.search(block)

            if not criteria_match:
                print(
                    f"Incomplete task {task_id}: "
                    "Acceptance criteria section was not found."
                )
                return False

            files_pattern = re.compile(
                r"(?ims)^\s*[-*+]\s+\*{0,2}\s*Files to create or modify"
                r"\s*\*{0,2}\s*:\s*\*{0,2}(.*?)"
                + section_end_pattern
            )
            files_match = files_pattern.search(block)
            file_paths = []

            if files_match:
                files_content = files_match.group(1)

                # Extract paths from inline code, including inline field values.
                file_paths.extend(
                    re.findall(r"`([^`]+)`", files_content)
                )

                # Extract unquoted paths from individual bullet lines.
                file_paths.extend(
                    re.findall(
                        r"(?m)^\s*[-*+]\s+([^\s`]+)\s*$",
                        files_content,
                    )
                )

                # Remove duplicates and ignore empty values.
                file_paths = list(
                    dict.fromkeys(
                        path.strip()
                        for path in file_paths
                        if path.strip()
                    )
                )
            def is_repository_relative_file(file_path: str) -> bool:
                normalized_path = file_path.replace("\\", "/")
                parts = normalized_path.split("/")
                root_file_names = {"dockerfile", "makefile", "license", "notice", "procfile"}
                has_file_name = (
                    len(parts) > 1
                    or bool(re.search(r"\.[A-Za-z0-9_-]+$", parts[-1]))
                    or parts[-1].casefold() in root_file_names
                )
                return (
                    has_file_name
                    and not normalized_path.startswith("/")
                    and not re.match(r"^[A-Za-z]:", normalized_path)
                    and all(part not in {"", ".", ".."} for part in parts)
                )

            has_relative_path = any(
                is_repository_relative_file(file_path)
                for file_path in file_paths
            )
            if not has_relative_path:
                print(
                    f"Incomplete task {task_id}: "
                    "no repository-relative file path found."
                )
                return False

        criteria_content = criteria_match.group(1).strip()

        criteria_bullets = re.findall(
            r"(?m)^\s*[-*+]\s+(.+?)\s*$",
            criteria_content,
        )

        # Accept inline acceptance criteria as well as bullet lists.
        # Example: "- **Acceptance criteria:** The model compiles."
        if not criteria_bullets and criteria_content:
            criteria_bullets = [criteria_content]

        if not criteria_bullets:
            print(
                f"Incomplete task {task_id}: "
                "Acceptance criteria contains no content."
            )
            return False

        return True


    def is_usable_plan(plan: str) -> bool:
        """Check that the plan meets the Blueprint Generator's contract."""

        return (
            bool(plan.strip())
            and has_valid_task_headings(plan)
            and not find_template_placeholders(plan)
            and has_complete_task_blocks(plan)
        )
    
    response = planner_llm.invoke(prompt)
    task_plan = extract_text(response)
    task_plan = normalize_task_headings(task_plan)
    print("\n===== ORIGINAL TASK PLAN =====\n")
    print(task_plan[:5000])
    print("\n===== END ORIGINAL TASK PLAN =====\n")
    if not task_plan:
        raise ValueError("Task Planner returned an empty implementation plan.")

    if not is_usable_plan(task_plan):
        print(
            "Task Planner output failed validation. "
            "Requesting one structured repair attempt..."
        )

        repair_prompt = f"""
You are repairing an invalid implementation plan for an AI software development system.

ORIGINAL USER REQUEST:
{user_request}

ANALYZED REQUIREMENTS:
{requirements}

APPROVED TECHNICAL DESIGN:
{design}

PREVIOUS PLAN THAT FAILED VALIDATION:
{task_plan}

CRITICAL REPAIR RULES:

1. Treat the ORIGINAL USER REQUEST, ANALYZED REQUIREMENTS,
   and APPROVED TECHNICAL DESIGN as the source of truth.

2. Do not preserve incorrect project details from the previous plan.
   The previous plan is untrusted because it failed validation.

3. Before writing tasks, identify the actual product, its required
   features, and the approved technology stack from the source inputs.

4. Do not replace the actual project with a generic user-profile,
   CRUD, or authentication application unless that is the actual
   requested project.

5. Every task heading must have its own Task ID field.
   Example:
   ### DB-001: Define the expense data model
   - Task ID: DB-001

6. Use the exact field labels required by the output contract.
   Do not substitute "File Paths" for "Files to create or modify".

7. Use only DB, BE, FE, and INT task prefixes.
   Testing tasks must use INT IDs.

8. Use the exact required section heading:
   ## 6. Integration and Testing Tasks

9. Before returning the plan, compare the summary and every task
   against the original inputs. Correct any project or technology
   mismatch before returning the result.

Your task is to rewrite the COMPLETE implementation plan for this exact project.
Do not merely explain the errors. Do not return a template or a list of suggested fixes.

NON-NEGOTIABLE OUTPUT CONTRACT:
- Return the eight sections specified below, in order.
- Every implementation task must have its own Markdown heading.
- Use only these heading forms:
  ### DB-001: Specific database task title
  ### BE-001: Specific backend task title
  ### FE-001: Specific frontend task title
  ### INT-001: Specific integration or testing task title
- Each task block must contain:
  - Task ID: the exact ID in its heading
  - Worker type: use a supported worker type
  - Objective: one concrete objective
  - Files to create or modify: repository-relative paths
  - Scope and implementation details: concrete task responsibilities
  - Dependencies (task IDs): declared IDs or None
  - Acceptance criteria: concrete, verifiable outcomes
- Use unique, sequential IDs within each category.
- Every dependency must refer to an existing task.
- Do not create dependency cycles.
- Every declared task must appear exactly once in Execution Order.
- Never represent tasks only as bullet points.
- Never include placeholders such as "[Insert project purpose]".
- Do not invent an architecture, technology, test framework, API, entity, or feature.
- Treat the approved technical design as authoritative for technology choices.
- If a detail is missing, explicitly state "Not specified" or record a clearly labelled assumption.
- Keep paths, entities, API contracts, task IDs, and dependencies consistent.
- Return only the corrected Markdown plan.

REQUIRED TOP-LEVEL SECTIONS:
# Implementation Plan
## 1. Project Summary
## 2. Project Structure
## 3. Database Tasks
## 4. Backend Tasks
## 5. Frontend Tasks
## 6. Integration and Testing Tasks
## 7. Execution Order
## 8. Handoff to the Project Blueprint Generator

Complete valid task block:
### DB-001: Define the task data model
- Task ID: DB-001
- Worker type: database
- Objective: Define the persistent task data model.
- Files to create or modify:
  - database/models/task.py
- Scope and implementation details:
  - Define the task fields supported by the requirements and design.
- Dependencies (task IDs): None
- Acceptance criteria:
  - The model contains the required fields.

Project-specific details must come from the original request, requirements, and approved design above.
Generate concrete tasks for the actual project, not generic tasks such as "implement backend API".
DEPENDENCY FIELD CORRECTION:

Inspect every task in the implementation plan.

Every task must contain exactly one dependency field:

- Dependencies (task IDs): None

OR

- Dependencies (task IDs): DB-001, BE-001

Do not omit this field, even if the task has no dependencies.

Do not change existing task IDs.
Do not invent new task IDs.
Preserve all existing implementation details.
Return the complete corrected implementation plan.
"""

        repair_response = planner_llm.invoke(repair_prompt)
        task_plan = extract_text(repair_response)
        task_plan = normalize_task_headings(task_plan)

    # Final gate: malformed output must never be sent to the Blueprint Generator.
    if not task_plan:
        raise ValueError(
            "Task Planner returned an empty plan after the repair attempt."
        )

    if not has_valid_task_headings(task_plan):
        raise ValueError(
            "Task Planner failed to produce parser-compatible task headings "
            "after the repair attempt. Expected headings such as "
            "'### DB-001: Define the database schema'. "
            "Inspect the model response before continuing."
        )

    found_placeholders = find_template_placeholders(task_plan)
    if found_placeholders:
        raise ValueError(
            "Task Planner still returned unresolved template placeholders "
            f"after the repair attempt: {found_placeholders}. "
            "The invalid plan was blocked before Blueprint generation."
        )
    if not has_complete_task_blocks(task_plan):
        # print("\n===== INVALID TASK PLAN =====\n")
        # print(task_plan)
        # print("\n===== END INVALID TASK PLAN =====\n")

        raise ValueError(
            "Task Planner validation failed. "
            "See the validation messages and plan printed above."
        )

    return task_plan
