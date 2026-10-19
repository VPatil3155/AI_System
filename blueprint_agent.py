from __future__ import annotations
import re
import json
from typing import Literal
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, model_validator, field_validator

load_dotenv()


# --------------------------------------------------
# 1. Blueprint schema
# --------------------------------------------------

class TechnologyStack(BaseModel):
    frontend: str = Field(description="Approved frontend technology, or 'Not applicable'")
    backend: str = Field(description="Approved backend technology, or 'Not applicable'")
    database: str = Field(description="Approved database technology, or 'Not applicable'")
    testing: list[str] = Field(default_factory=list)
    other: list[str] = Field(default_factory=list)


class BlueprintFile(BaseModel):
    path: str = Field(description="Repository-relative path; no absolute paths")
    purpose: str = Field(description="Responsibility of this file")
    owner: Literal["frontend", "backend", "database", "shared", "testing", "configuration"]
    task_ids: list[str] = Field(default_factory=list, description="Task IDs from the Task Planner")
    depends_on: list[str] = Field(default_factory=list, description="File paths this file relies on")


class BlueprintEntity(BaseModel):
    name: str
    description: str = ""
    fields: list[str] = Field(default_factory=list, description="Only fields supported by design/requirements")
    relationships: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class BlueprintEndpoint(BaseModel):
    method: str
    path: str
    purpose: str
    request_contract: str = ""
    response_contract: str = ""
    auth_required: bool | None = None
    related_entities: list[str] = Field(default_factory=list)


class BlueprintInterface(BaseModel):
    name: str
    producer: str
    consumer: str
    contract: str
    related_files: list[str] = Field(default_factory=list)


class BlueprintTask(BaseModel):
    task_id: str
    title: str
    worker_type: Literal["frontend", "backend", "database", "testing", "shared", "configuration"]
    objective: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list, description="Existing task IDs")
    acceptance_criteria: list[str] = Field(default_factory=list)




def generate_directory_tree(project_name: str, file_paths: list[str]) -> str:
    """Generate a deterministic directory tree from repository-relative paths."""
    root = {"dirs": {}, "files": set()}

    for file_path in file_paths:
        normalized_path = file_path.replace("\\", "/").strip("/")
        parts = normalized_path.split("/")

        if not normalized_path or any(part in ("", ".", "..") for part in parts):
            raise ValueError(f"Invalid repository-relative path: {file_path!r}")

        current = root

        for directory in parts[:-1]:
            current = current["dirs"].setdefault(
                directory,
                {"dirs": {}, "files": set()},
            )

        current["files"].add(parts[-1])

    lines = [f"{project_name}/"]

    def render(node: dict, prefix: str = "") -> None:
        entries = []

        for name, child in node["dirs"].items():
            entries.append((name, True, child))

        for name in node["files"]:
            entries.append((name, False, None))

        entries.sort(key=lambda entry: (not entry[1], entry[0].lower()))

        for index, (name, is_directory, child) in enumerate(entries):
            is_last = index == len(entries) - 1
            connector = "└── " if is_last else "├── "
            lines.append(
                f"{prefix}{connector}{name}{'/' if is_directory else ''}"
            )

            if is_directory:
                next_prefix = prefix + ("    " if is_last else "│   ")
                render(child, next_prefix)

    render(root)
    return "\n".join(lines)




class ProjectBlueprint(BaseModel):
    blueprint_version: str = "1.0"
    project_name: str
    project_summary: str
    approved_stack: TechnologyStack
    architecture_style: str
    directory_tree: str = Field(description="Plain-text repository tree")
    files: list[BlueprintFile]
    entities: list[BlueprintEntity] = Field(default_factory=list)
    api_endpoints: list[BlueprintEndpoint] = Field(default_factory=list)
    interfaces: list[BlueprintInterface] = Field(default_factory=list)
    environment_variables: list[str] = Field(default_factory=list, description="Names only; never values or secrets")
    install_commands: list[str] = Field(default_factory=list)
    run_commands: list[str] = Field(default_factory=list)
    test_commands: list[str] = Field(default_factory=list)
    tasks: list[BlueprintTask]
    assumptions: list[str] = Field(default_factory=list)
    unresolved_decisions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_blueprint_consistency(self):
        print("DEBUG: Blueprint consistency validator is running")
        # 1. Validate file paths
        file_paths = [item.path for item in self.files]

        if len(file_paths) != len(set(file_paths)):
            raise ValueError("Blueprint contains duplicate file paths.")

        for item in self.files:
            path = item.path.replace("\\", "/")

            if (
                not path.strip()
                or path.startswith("/")
                or path.startswith("//")
                or (len(path) >= 2 and path[1] == ":")
                or ".." in path.split("/")
                or "." in path.split("/")
            ):
                raise ValueError(
                    f"Unsafe repository-relative path: {item.path!r}"
                )

        known_paths = set(file_paths)

        # 2. Validate task IDs
        task_ids = [task.task_id for task in self.tasks]

        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Blueprint contains duplicate task IDs.")

        known_tasks = set(task_ids)

        # 3. Validate task dependencies and file mappings
        for task in self.tasks:
            missing_dependencies = set(task.dependencies) - known_tasks

            if missing_dependencies:
                raise ValueError(
                    f"Task {task.task_id} references unknown dependencies: "
                    f"{sorted(missing_dependencies)}"
                )

            missing_files = set(task.files) - known_paths

            if missing_files:
                raise ValueError(
                    f"Task {task.task_id} references files absent from blueprint: "
                    f"{sorted(missing_files)}"
                )

            if not task.files:
                raise ValueError(
                    f"Task {task.task_id} has no file mappings."
                )

            if task.task_id in task.dependencies:
                raise ValueError(
                    f"Task {task.task_id} cannot depend on itself."
                )

        # # 4. Ensure every task is mapped back to its files
        # for item in self.files:
        #     unknown_task_ids = set(item.task_ids) - known_tasks

        #     if unknown_task_ids:
        #         raise ValueError(
        #             f"File {item.path} references unknown task IDs: "
        #             f"{sorted(unknown_task_ids)}"
        #         )

        #     for task_id in item.task_ids:
        #         task = next(
        #             task for task in self.tasks
        #             if task.task_id == task_id
        #         )

        #         if item.path not in task.files:
        #             raise ValueError(
        #                 f"File {item.path} lists task {task_id}, "
        #                 "but that task does not list this file."
        #             )

        # 5. Ensure task dependencies contain no cycles
        dependency_map = {
            task.task_id: task.dependencies
            for task in self.tasks
        }

        visited = set()
        active = set()

        def visit(task_id: str):
            if task_id in active:
                raise ValueError(
                    f"Circular task dependency detected at {task_id}."
                )

            if task_id in visited:
                return

            active.add(task_id)

            for dependency in dependency_map[task_id]:
                visit(dependency)

            active.remove(task_id)
            visited.add(task_id)

        for task_id in dependency_map:
            visit(task_id)

        # 6. Validate file dependencies
        for item in self.files:
            missing_dependencies = set(item.depends_on) - known_paths

            if missing_dependencies:
                raise ValueError(
                    f"File {item.path} references unknown file dependencies: "
                    f"{sorted(missing_dependencies)}"
                )

        #         # 7. Validate the directory tree
        # expected_tree = generate_directory_tree(
        #     self.project_name,
        #     file_paths,
        # )

        # if self.directory_tree.strip() != expected_tree.strip():
        #     raise ValueError(
        #         "Directory tree does not match the file inventory. "
        #         "Generate the tree using generate_directory_tree()."
        #     )

        # 8. Validate architecture decision
        if not self.architecture_style.strip():
            raise ValueError("Architecture style cannot be empty.")

        return self

    @field_validator("project_name", mode="before")
    @classmethod
    def normalize_project_name(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Project name must be a string.")

        # Convert a human-readable name into a directory-safe slug.
        value = value.strip().lower()
        value = re.sub(r"\s+", "-", value)
        value = re.sub(r"[^a-z0-9._-]+", "-", value)
        value = re.sub(r"-+", "-", value)
        value = value.strip("-.")

        if not value or value in {".", ".."}:
            raise ValueError("Invalid project name.")

        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", value):
            raise ValueError(f"Invalid project name: {value!r}")

        return value

# --------------------------------------------------
# 2. LLM configuration
# --------------------------------------------------

blueprint_llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)

groq_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)




def extract_task_ids_and_dependencies(
    task_plan: str,
) -> dict[str, list[str]]:
    """Extract every declared task ID and its section's dependencies."""

    task_dependencies: dict[str, list[str]] = {}
    current_task_ids: list[str] = []

    task_id_pattern = re.compile(
        r"\b(?:DB|BE|FE|INT|TEST)-\d+\b",
        re.IGNORECASE,
    )

    dependency_pattern = re.compile(
        r"Dependencies(?:\s*\([^)]*\))?\s*:\s*(.*)",
        re.IGNORECASE,
    )

    for line in task_plan.splitlines():
        # Only treat IDs on a Task ID declaration as task definitions.
        if re.search(r"\bTask ID\s*:", line, re.IGNORECASE):
            declared_ids = task_id_pattern.findall(
                line.split(":", 1)[1]
            )

            current_task_ids = [
                task_id.upper() for task_id in declared_ids
            ]

            for task_id in current_task_ids:
                task_dependencies.setdefault(task_id, [])

            continue

        dependency_match = dependency_pattern.search(line)

        if dependency_match and current_task_ids:
            dependency_text = dependency_match.group(1)

            dependencies = [
                task_id.upper()
                for task_id in task_id_pattern.findall(dependency_text)
            ]

            for task_id in current_task_ids:
                task_dependencies[task_id] = dependencies

    return task_dependencies


def validate_task_plan_preservation(
    expected_tasks: dict[str, list[str]],
    actual_tasks: dict[str, list[str]],
) -> None:
    """Reject blueprint task IDs or dependencies that differ from the plan."""
    if set(actual_tasks) != set(expected_tasks):
        raise ValueError(
            "Blueprint task IDs differ from the original Task Planner output. "
            f"Expected: {sorted(expected_tasks)}; "
            f"received: {sorted(actual_tasks)}"
        )

    for task_id, expected_dependencies in expected_tasks.items():
        if actual_tasks[task_id] != expected_dependencies:
            raise ValueError(
                f"Dependencies for {task_id} differ from the Task Planner. "
                f"Expected: {expected_dependencies}; "
                f"received: {actual_tasks[task_id]}"
            )


def parse_canonical_tasks(task_plan: str) -> list[dict]:
    """Parse individual task blocks from Task Planner Markdown output."""

    task_pattern = re.compile(
        r"^###\s+((?:DB|BE|FE|INT|TEST)-\d+):\s*(.+?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    task_id_pattern = re.compile(
        r"\b(?:DB|BE|FE|INT|TEST)-\d+\b",
        re.IGNORECASE,
    )

    matches = list(task_pattern.finditer(task_plan))

    if not matches:
        raise ValueError(
            "No individual task headings found. Expected headings such as "
            "'### DB-001: Define the database schema'."
        )

    def read_field(block: str, label: str) -> str:
        pattern = (
            rf"(?im)^\s*[-*+]\s*{re.escape(label)}\s*:\s*(.*)$"
        )
        match = re.search(pattern, block)
        return match.group(1).strip() if match else ""

    def read_section(block: str, label: str) -> str:
        pattern = (
            rf"(?ims)^\s*[-*+]\s*{re.escape(label)}\s*:\s*(.*?)"
            rf"(?=^\s*[-*+]\s*[A-Za-z][^:\n]*\s*:|\Z)"
        )
        match = re.search(pattern, block)
        return match.group(1) if match else ""

    def read_bullets(section: str) -> list[str]:
        items = re.findall(
            r"(?m)^\s*[-*+]\s+(.+?)\s*$",
            section,
        )
        return [
            item.strip().strip("`")
            for item in items
            if item.strip().lower() not in {"none", "n/a"}
        ]

    tasks = []
    seen_ids = set()

    for index, match in enumerate(matches):
        task_id = match.group(1).upper()
        title = match.group(2).strip()

        if task_id in seen_ids:
            raise ValueError(f"Duplicate task ID: {task_id}")
        seen_ids.add(task_id)

        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(task_plan)
        )
        block = task_plan[match.end():end]

        objective = read_field(block, "Objective")

        dependency_text = (
            read_field(block, "Dependencies (task IDs)")
            or read_field(block, "Dependencies")
        )
        dependencies = [
            value.upper()
            for value in task_id_pattern.findall(dependency_text)
        ]

        files = read_bullets(
            read_section(block, "Files to create or modify")
        )
        criteria = read_bullets(
            read_section(block, "Acceptance criteria")
        )

        if not objective:
            raise ValueError(
                f"{task_id} has no Objective. Check its Task Planner block."
            )
        if not files:
            raise ValueError(
                f"{task_id} has no files under 'Files to create or modify'."
            )
        if not criteria:
            raise ValueError(
                f"{task_id} has no Acceptance criteria."
            )

        prefix = task_id.split("-")[0]
        worker_type = {
            "DB": "database",
            "BE": "backend",
            "FE": "frontend",
            "INT": "testing",
            "TEST": "testing",
        }[prefix]

        tasks.append({
            "task_id": task_id,
            "title": title,
            "worker_type": worker_type,
            "objective": objective,
            "files": files,
            "dependencies": dependencies,
            "acceptance_criteria": criteria,
        })

    all_ids = {task["task_id"] for task in tasks}

    for task in tasks:
        unknown = set(task["dependencies"]) - all_ids
        if unknown:
            raise ValueError(
                f"{task['task_id']} references undefined dependencies: "
                f"{sorted(unknown)}"
            )

    return tasks



def normalize_blueprint_task_dependencies(
    blueprint_data: dict,
    task_plan: str,
) -> dict:
    """
    Preserve canonical Task Planner tasks and merge Blueprint LLM file mappings.
    """
    canonical_tasks = parse_canonical_tasks(task_plan)
    llm_tasks = blueprint_data.get("tasks", [])
    blueprint_files = blueprint_data.setdefault("files", [])

    # Index LLM tasks.
    llm_tasks_by_id = {}

    for task in llm_tasks:
        task_id = str(task.get("task_id", "")).strip().upper()

        if not task_id:
            continue

        if task_id in llm_tasks_by_id:
            raise ValueError(
                f"Blueprint LLM returned duplicate task ID: {task_id}"
            )

        llm_tasks_by_id[task_id] = task

    # Reject task IDs that do not exist in the Task Planner output.
    canonical_ids = {task["task_id"] for task in canonical_tasks}
    unexpected_ids = set(llm_tasks_by_id) - canonical_ids

    if unexpected_ids:
        raise ValueError(
            "Blueprint LLM returned task IDs not present in the Task Planner: "
            f"{sorted(unexpected_ids)}"
        )

    # Normalize and index the generated file inventory.
    files_by_path = {}

    for item in blueprint_files:
        raw_path = item.get("path", "")

        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("Blueprint contains an empty or invalid file path.")

        path = raw_path.replace("\\", "/").strip("/")

        if path in files_by_path:
            raise ValueError(
                f"Blueprint contains duplicate file path: {path}"
            )

        item["path"] = path
        files_by_path[path] = item

    final_tasks = []

    for canonical in canonical_tasks:
        task_id = canonical["task_id"]
        generated = llm_tasks_by_id.get(task_id, {})

        generated_files = generated.get("files", [])
        if not isinstance(generated_files, list):
            generated_files = []

        # The Task Planner's file mappings are authoritative.
        # Additional LLM file mappings are retained.
        task_files = list(dict.fromkeys(
            [
                path.replace("\\", "/").strip("/")
                for path in canonical["files"]
                if isinstance(path, str) and path.strip()
            ]
            + [
                path.replace("\\", "/").strip("/")
                for path in generated_files
                if isinstance(path, str) and path.strip()
            ]
        ))

        if not task_files:
            raise ValueError(
                f"{task_id} has no usable file mappings. "
                "Every Task Planner task must specify at least one file."
            )

        # Keep task identity, title, objective, dependencies, and criteria
        # from the canonical Task Planner output.
        final_tasks.append({
            **canonical,
            "files": task_files,
        })

        # Add missing task files to the blueprint inventory.
        for path in task_files:
            if path not in files_by_path:
                files_by_path[path] = {
                    "path": path,
                    "purpose": (
                        f"Implementation file for {task_id}: "
                        f"{canonical['objective']}"
                    ),
                    "owner": canonical["worker_type"],
                    "task_ids": [],
                    "depends_on": [],
                }

    # Rebuild reverse file-to-task mappings.
    task_ids_by_file = {}

    for task in final_tasks:
        for path in task["files"]:
            task_ids_by_file.setdefault(path, []).append(task["task_id"])

    for path, item in files_by_path.items():
        item["task_ids"] = sorted(
            set(task_ids_by_file.get(path, []))
        )

    blueprint_data["files"] = list(files_by_path.values())
    blueprint_data["tasks"] = final_tasks

    return blueprint_data

# --------------------------------------------------
# 3. Blueprint Agent
# --------------------------------------------------

def create_project_blueprint(
    user_request: str,
    requirements: str,
    design: str,
    task_plan: str,
) -> ProjectBlueprint:
    """
    Convert approved requirements, design and task plan into a validated
    machine-readable project blueprint.

    This function does not create or modify files. File creation belongs to
    the separate Scaffolder component.
    """
    prompt = f"""
You are the Project Blueprint Agent in an AI software development pipeline.

Produce a precise project blueprint that downstream scaffolding and coding
agents can execute. Return data matching the provided ProjectBlueprint schema.

SOURCE OF TRUTH:
- The approved technical design determines the architecture and technology stack.
- Requirements determine the required behavior.
- The task plan determines implementation task IDs, task dependencies and work scope.
- Do not introduce new features, frameworks, databases, services, or deployment
  platforms that are not supported by these inputs.
- If a detail is missing, record it in assumptions or unresolved_decisions.

USER REQUEST:
{user_request}

APPROVED REQUIREMENTS:
{requirements}

APPROVED TECHNICAL DESIGN:
{design}

TASK PLANNER OUTPUT:
{task_plan}


BLUEPRINT RULES:

1. Create a practical repository-relative file inventory. Give each file
   one clear purpose and one owner.

2. Preserve every task ID and dependency from the Task Planner exactly.
   Do not omit tasks or invent dependencies.

3. Every task MUST have a non-empty files list containing at least one
   path from the files inventory.

4. Task-to-file mappings MUST be bidirectional:
   - Every path in task.files must exist in files.
   - If a file lists a task ID in task_ids, that task must list the
     same file path in its files list.
   - Use identical path strings everywhere.

5. Map tasks according to their objectives:
   - Database tasks own database models or schema files.
   - Backend tasks own API routes, services and backend logic.
   - Frontend tasks own UI components and frontend logic.
   - Testing tasks own test files.
   - Configuration tasks own relevant configuration and documentation.

6. The directory_tree MUST contain the complete directory hierarchy
   and every path listed in files. Do not return only the project name.

7. Generate directory_tree as a complete plain-text directory tree.

   The tree must show the root project directory, every intermediate
   directory, and every file listed in files.

   Example:

   task-management-application/
   ├── database/
   │   └── models/
   │       └── task.py
   ├── backend/
   │   └── api/
   │       └── routes/
   │           └── task_routes.py
   ├── frontend/
   │   └── components/
   │       └── TaskList.js
   └── tests/
       └── test_task_crud.py

   Include every file from the files inventory, not just the example.
   Use a relative root directory name without a leading slash.


8. Include entities, API endpoints and interfaces when supported by
   the requirements and approved design. Keep names and contracts
   consistent throughout the blueprint.

9. List environment variable names only. Never include secret values,
   passwords, tokens or API keys.

10. Include install, run and test commands only when they can be
    reasonably determined from the approved stack.

11. Use safe repository-relative paths. Never use absolute paths,
    drive-letter paths or parent-directory traversal.

12. Do not generate source code or create files. The Scaffolder will
    create the initial project structure after validation.

13. Keep the blueprint practical and minimal. Do not add unsupported
    features or unnecessary files.

    
14. Treat the Task Planner output as the authoritative source for task IDs
    and dependencies.

15. Copy each task's dependency list exactly from the Task Planner output.
    Do not add, remove, rename, or infer task dependencies.

16. Never invent task IDs such as DB-000, BE-000, FE-000, or TEST-000.
    Every task ID and dependency must already exist in the Task Planner output.

17. If the Task Planner output does not explicitly provide dependencies
    for a task, use an empty dependency list rather than guessing.

18. Do not create additional tasks unless they already exist in the
    Task Planner output.

    

ARCHITECTURE RULE:
- Use only the architecture style explicitly specified in the approved design.
- Do not infer Monolithic, Microservices, MVC, or another architecture
  from the technology stack.
- If the design does not specify an architecture style, return "Not specified".

# COMMAND RULES:
# - Do not invent application entry points or package scripts.
# - Every run command must match the planned file inventory and configuration.
# - If the backend entry point is unknown, record the decision in unresolved_decisions.
# - If the frontend package manager or run script is unknown, do not invent
#   a frontend run command.
# - Do not claim commands are verified unless their required files and scripts exist.


COMMAND VALIDATION:
- The approved technology stack is the source of truth.
- Do not assume that npm is the package manager unless the frontend
  configuration supports it.
- Include npm scripts only when package.json and the relevant scripts
  are explicitly included in the planned file inventory or design.
- For a FastAPI backend, include Python installation or execution
  commands only when the dependency and entry-point configuration
  is specified.
- Do not invent backend entry points, package scripts, or test commands.
- Do not claim that any command has been verified unless its required
  files and configuration are present in the blueprint.
- If a command cannot be determined, omit it and record the missing
  information in unresolved_decisions.


TASK DEPENDENCY RULES:
- Preserve the dependencies from the original Task Planner output.
- A task may depend only on tasks that must finish before it starts.
- Database/model tasks must not depend on backend API tasks when those APIs depend on the database model.
- Never create circular dependencies.
- If task A depends on task B, task B must not depend on task A.
- Use an empty dependencies list when a task has no prerequisites.


CRITICAL TASK PRESERVATION RULES:

1. Use the original task_plan as the authoritative source for tasks.
2. Preserve every original task_id exactly. Never invent, rename, or add IDs.
3. Preserve every original dependency exactly.
4. Never create dependencies such as DB-000 unless that exact ID exists in task_plan.
5. Every dependency must refer to an existing task_id.
6. Do not add extra tasks or dependencies.
7. If task_plan is empty or missing, report the unresolved decision rather than inventing task IDs.

CRITICAL CONSISTENCY RULES:
1. Treat the supplied task plan and approved technical design as authoritative.
2. Preserve every task ID and its dependency list exactly as specified in the task plan.
3. Never invent dependency IDs. Every dependency must reference an existing task ID.
4. A task with no dependencies in the task plan must have an empty dependencies list.
5. Preserve the approved frontend, backend, database, and testing technologies.
   Never substitute PostgreSQL for MongoDB, or vice versa, unless explicitly
   authorized by the supplied design and requirements.
6. Do not invent an architecture style. Use the supplied design's architecture
   or "Not specified" if it is not established.
7. Ensure all required fields in every task and file satisfy the ProjectBlueprint schema.

project_name must be a filesystem-safe slug:
- lowercase letters and digits
- hyphens between words
- no spaces
- no directory separators
Example: "Task Management Application" becomes
"task-management-application".

"""

    
    # # structured_llm = blueprint_llm.with_structured_output(ProjectBlueprint)
    # structured_llm = groq_llm.with_structured_output(ProjectBlueprint)
    # blueprint = structured_llm.invoke(prompt)

    

    # Generate the blueprint using the local Ollama model

    # Ask Ollama to generate JSON instead of free-form text.
   
    # Generate structured JSON without validating it as ProjectBlueprint yet.
    # This lets us normalize dependencies before Pydantic validation.
    structured_llm = blueprint_llm.with_structured_output(
        ProjectBlueprint.model_json_schema(),
        method="json_schema",
    )

    blueprint_data = structured_llm.invoke(prompt)

    if not isinstance(blueprint_data, dict):
        raise TypeError(
            "Blueprint Agent did not return a JSON object."
        )

    # Correct task dependencies using the authoritative Task Planner output.
    blueprint_data = normalize_blueprint_task_dependencies(
        blueprint_data,
        task_plan,
    )

    # Validate only after normalization.
    blueprint = ProjectBlueprint.model_validate(blueprint_data)





    # Commands are included only when the blueprint has enough
    # configuration to support them.
    planned_paths = {
        item.path.replace("\\", "/").lstrip("./")
        for item in blueprint.files
    }

    has_python_config = any(
        path in planned_paths
        for path in ("pyproject.toml", "requirements.txt")
    )

    has_package_json = any(
        path == "package.json" or path.endswith("/package.json")
        for path in planned_paths
    )

    if not has_python_config:
        blueprint.install_commands = [
            command
            for command in blueprint.install_commands
            if not command.strip().startswith(("pip ", "uv "))
        ]

        blueprint.run_commands = [
            command
            for command in blueprint.run_commands
            if not command.strip().startswith(("fastapi ", "uvicorn "))
        ]

        blueprint.test_commands = [
            command
            for command in blueprint.test_commands
            if not command.strip().startswith(("pytest", "python -m pytest"))
        ]

    if not has_package_json:
        blueprint.install_commands = [
            command
            for command in blueprint.install_commands
            if not command.strip().startswith("npm ")
        ]

        blueprint.run_commands = [
            command
            for command in blueprint.run_commands
            if not command.strip().startswith("npm ")
        ]

        blueprint.test_commands = [
            command
            for command in blueprint.test_commands
            if not command.strip().startswith("npm ")
        ]

    
    expected_tasks = extract_task_ids_and_dependencies(task_plan)

    actual_tasks = {
        task.task_id: task.dependencies
        for task in blueprint.tasks
    }

    if expected_tasks:
        validate_task_plan_preservation(expected_tasks, actual_tasks)




    
    # Verify task IDs and dependencies against the original task plan.
    expected_tasks = extract_task_ids_and_dependencies(task_plan)

    if expected_tasks:
        actual_tasks = {
            task.task_id: task.dependencies
            for task in blueprint.tasks
        }

        if set(actual_tasks) != set(expected_tasks):
            raise ValueError(
                "Blueprint task IDs differ from the original Task Planner output. "
                f"Expected: {sorted(expected_tasks)}; "
                f"received: {sorted(actual_tasks)}"
            )

        for task_id, expected_dependencies in expected_tasks.items():
            if actual_tasks[task_id] != expected_dependencies:
                raise ValueError(
                    f"Dependencies for {task_id} differ from the Task Planner. "
                    f"Expected: {expected_dependencies}; "
                    f"received: {actual_tasks[task_id]}"
                )

        
    # Validate commands against the planned file inventory.
    planned_paths = {
        item.path.replace("\\", "/").lstrip("./")
        for item in blueprint.files
    }

    # Remove npm commands if package.json is not planned.
    if "package.json" not in planned_paths:
        blueprint.install_commands = [
            command for command in blueprint.install_commands
            if not command.strip().startswith("npm ")
        ]
        blueprint.run_commands = [
            command for command in blueprint.run_commands
            if not command.strip().startswith("npm ")
        ]
        blueprint.test_commands = [
            command for command in blueprint.test_commands
            if not command.strip().startswith("npm ")
        ]

    
    # Remove Uvicorn commands when their referenced app module is not planned.
    planned_path_set = {
        path.replace("\\", "/").lstrip("./")
        for path in planned_paths
    }

    if "backend.py" not in planned_path_set and "backend/__init__.py" not in planned_path_set:
        blueprint.run_commands = [
            command
            for command in blueprint.run_commands
            if not command.strip().startswith("uvicorn backend:")
        ]
        blueprint.unresolved_decisions.append(
            "FastAPI application entry point must be specified before a run command can be generated."
        )
    


    if not isinstance(blueprint, ProjectBlueprint):
        # Defensive fallback for model/provider integrations that return a dict.
        if isinstance(blueprint, dict):
            blueprint = ProjectBlueprint.model_validate(blueprint)
        else:
            raise TypeError(
                "Blueprint Agent did not return a ProjectBlueprint. "
                f"Received: {type(blueprint).__name__}"
            )

    # 1. Build file-to-task mappings.
    task_ids_by_file = {}

    for task in blueprint.tasks:
        for file_path in task.files:
            task_ids_by_file.setdefault(file_path, []).append(task.task_id)

    # 2. Populate task_ids for each file.
    for blueprint_file in blueprint.files:
        blueprint_file.task_ids = task_ids_by_file.get(
            blueprint_file.path, []
        )

    # 3. Generate the directory tree deterministically.
    blueprint.directory_tree = generate_directory_tree(
        blueprint.project_name,
        [item.path for item in blueprint.files],
    )

    return blueprint





def blueprint_to_json(blueprint: ProjectBlueprint) -> str:
    """Serialize a blueprint with a freshly generated directory tree."""
    data = blueprint.model_dump(mode="json")

    # Always rebuild the tree from the file inventory.
    data["directory_tree"] = generate_directory_tree(
        data["project_name"],
        [item["path"] for item in data["files"]],
    )

    # Return the normalized data as JSON.
    return json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
    )

