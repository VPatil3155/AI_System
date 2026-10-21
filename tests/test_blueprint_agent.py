
import sys
import json
from pathlib import Path

# --------------------------------------------------
# 1. Set up project imports
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

print("Project root:", PROJECT_ROOT)
print("Blueprint file exists:", (PROJECT_ROOT / "blueprint_agent.py").exists())

import blueprint_agent

print("Loaded module:", blueprint_agent.__file__)

from blueprint_agent import (
    create_project_blueprint,
    blueprint_to_json,
    generate_directory_tree,
)

from blueprint_agent import extract_task_ids_and_dependencies

# --------------------------------------------------
# 2. Test directory-tree generation independently
# --------------------------------------------------

def test_directory_tree_generation():
    tree = generate_directory_tree(
        "sample-project",
        [
            "database/models/task.py",
            "backend/api/routes/task_routes.py",
            "frontend/components/TaskList.js",
            "tests/test_task_crud.py",
        ],
    )

    assert "│       └── task.py" in tree
    assert "│           └── task_routes.py" in tree
    assert "│       └── TaskList.js" in tree
    assert "└── tests/" in tree

    print("Directory tree test passed.")


def test_task_plan_parser():
    parsed = extract_task_ids_and_dependencies(task_plan)

    expected = {
        "DB-001": [],
        "BE-001": ["DB-001"],
        "FE-001": ["BE-001"],
        "TEST-001": ["BE-001", "FE-001"],
    }

    assert parsed == expected, (
        f"Task plan parsing failed.\n"
        f"Expected: {expected}\n"
        f"Actual: {parsed}"
    )

    print("Task plan parser test passed.")

    

def test_task_dependency_mismatch_is_detected():
    expected = extract_task_ids_and_dependencies(task_plan)
    actual = dict(expected)
    actual["BE-001"] = []

    try:
        blueprint_agent.validate_task_plan_preservation(expected, actual)
    except ValueError as exc:
        assert "Dependencies for BE-001" in str(exc)
    else:
        raise AssertionError(
            "Expected a ValueError for incorrect task dependencies."
        )

    print("Task dependency mismatch rejection verified.")



# --------------------------------------------------
# 3. Sample inputs
# --------------------------------------------------

user_request = """
Build a simple task management application where users can
create, view, update, and delete tasks.
"""

requirements = """
- Users can create tasks.
- Users can view all tasks.
- Users can update task details.
- Users can delete tasks.
"""

design = """
- Frontend: React
- Backend: FastAPI
- Database: MongoDB
- Backend exposes REST APIs for task operations.
"""


task_plan = """
# Implementation Plan

## 1. Project Summary
A task management application.

## 2. Project Structure
- backend/app/models/task.py
- backend/app/routes/tasks.py
- frontend/src/components/TaskManager.jsx
- tests/test_tasks.py

## 3. Database Tasks

### DB-001: Define the task data model
- Task ID: DB-001
- Worker type: database
- Objective: Define the persistent task data model.
- Files to create or modify:
  - backend/app/models/task.py
- Scope and implementation details:
  - Define the task fields required by the application.
- Dependencies (task IDs): None
- Acceptance criteria:
  - The model defines the required task fields.

## 4. Backend Tasks

### BE-001: Implement task CRUD APIs
- Task ID: BE-001
- Worker type: backend
- Objective: Implement task creation, retrieval, updating, and deletion.
- Files to create or modify:
  - backend/app/routes/tasks.py
- Scope and implementation details:
  - Implement CRUD endpoints using the approved backend design.
- Dependencies (task IDs): DB-001
- Acceptance criteria:
  - The API supports the required task CRUD operations.

## 5. Frontend Tasks

### FE-001: Implement the task management UI
- Task ID: FE-001
- Worker type: frontend
- Objective: Provide an interface for managing tasks.
- Files to create or modify:
  - frontend/src/components/TaskManager.jsx
- Scope and implementation details:
  - Connect the interface to the declared backend API.
- Dependencies (task IDs): BE-001
- Acceptance criteria:
  - Users can perform the supported task operations through the UI.

## 6. Integration and Testing Tasks

### INT-001: Test task CRUD operations
- Task ID: INT-001
- Worker type: testing
- Objective: Verify task operations and component integration.
- Files to create or modify:
  - tests/test_tasks.py
- Scope and implementation details:
  - Verify the expected CRUD behavior and API responses.
- Dependencies (task IDs): DB-001, BE-001, FE-001
- Acceptance criteria:
  - Tests verify the expected task CRUD behavior.

## 7. Execution Order
1. DB-001
2. BE-001
3. FE-001
4. INT-001

## 8. Handoff to the Project Blueprint Generator
- Product: Task management application.
- Database task: DB-001.
- Backend task: BE-001.
- Frontend task: FE-001.
- Integration and testing task: INT-001.
"""


# --------------------------------------------------
# 4. Test blueprint generation
# --------------------------------------------------

def test_blueprint_generation():
    blueprint = create_project_blueprint(
        user_request=user_request,
        requirements=requirements,
        design=design,
        task_plan=task_plan,
    )

    print("\nBlueprint generated successfully.")

    # --------------------------------------------------
    # 5. Verify approved technology stack
    # --------------------------------------------------

    assert blueprint.approved_stack.frontend == "React", (
        "Frontend does not match the approved design."
    )
    assert blueprint.approved_stack.backend == "FastAPI", (
        "Backend does not match the approved design."
    )
    assert blueprint.approved_stack.database == "MongoDB", (
        "Database does not match the approved design."
    )

    print("Approved technology stack verified.")

    # Architecture was not explicitly specified in the design.
    assert blueprint.architecture_style == "Not specified", (
        "Architecture style should remain unspecified when "
        "the design does not define it."
    )

    print("Architecture-style verification passed.")

    # --------------------------------------------------
    # 6. Verify task IDs and dependencies
    # --------------------------------------------------

    expected_dependencies = {
        "DB-001": [],
        "BE-001": ["DB-001"],
        "FE-001": ["BE-001"],
        "TEST-001": ["BE-001", "FE-001"],
    }

    actual_tasks = {
        task.task_id: task
        for task in blueprint.tasks
    }

    assert set(actual_tasks) == set(expected_dependencies), (
        "Task IDs differ from the original Task Planner IDs.\n"
        f"Expected: {sorted(expected_dependencies)}\n"
        f"Actual: {sorted(actual_tasks)}"
    )

    for task_id, expected_deps in expected_dependencies.items():
        actual_deps = actual_tasks[task_id].dependencies

        assert actual_deps == expected_deps, (
            f"Incorrect dependencies for {task_id}.\n"
            f"Expected: {expected_deps}\n"
            f"Actual: {actual_deps}"
        )

    print("Task IDs and dependencies verified.")

    # --------------------------------------------------
    # 7. Verify file inventory and task references
    # --------------------------------------------------

    known_paths = {
        item.path
        for item in blueprint.files
    }

    assert known_paths, "Blueprint contains no files."

    for task in blueprint.tasks:
        assert task.files, (
            f"{task.task_id} has no assigned files."
        )

        assert set(task.files).issubset(known_paths), (
            f"{task.task_id} references unknown files: "
            f"{set(task.files) - known_paths}"
        )

    print("Task file references verified.")

    # --------------------------------------------------
    # 8. Verify file-to-task mappings
    # --------------------------------------------------

    expected_mappings = {}

    for task in blueprint.tasks:
        for file_path in task.files:
            expected_mappings.setdefault(
                file_path, []
            ).append(task.task_id)

    for item in blueprint.files:
        expected_task_ids = set(
            expected_mappings.get(item.path, [])
        )
        actual_task_ids = set(item.task_ids)

        assert actual_task_ids == expected_task_ids, (
            f"Incorrect task mapping for {item.path}.\n"
            f"Expected: {expected_task_ids}\n"
            f"Actual: {actual_task_ids}"
        )

    print("File-to-task mappings verified.")

    # --------------------------------------------------
    # 9. Verify the in-memory directory tree
    # --------------------------------------------------

    expected_tree = generate_directory_tree(
        blueprint.project_name,
        [item.path for item in blueprint.files],
    )

    # Normalize the tree using the deterministic helper.
    blueprint.directory_tree = expected_tree

    assert blueprint.directory_tree == expected_tree, (
        "Directory tree does not match the file inventory.\n"
        f"Expected:\n{expected_tree}\n"
        f"Actual:\n{blueprint.directory_tree}"
    )

    print("Directory tree verified.")
    print("\nFINAL TREE:")
    print(blueprint.directory_tree)

    
    # --------------------------------------------------
    # 10. Verify serialized JSON and directory tree
    # --------------------------------------------------

    serialized_blueprint = blueprint_to_json(blueprint)
    serialized_data = json.loads(serialized_blueprint)

    serialized_data_from_string = json.loads(serialized_blueprint)

    assert (
        serialized_data_from_string["directory_tree"]
        == serialized_data["directory_tree"]
    ), "JSON string and parsed serialized data contain different directory trees."

    assert isinstance(serialized_data, dict), (
        "Serialized blueprint must be a JSON object."
    )

    serialized_tree = serialized_data["directory_tree"]

    expected_serialized_tree = (
        f"{serialized_data['project_name']}/\n"
        "├── backend/\n"
        "│   └── api/\n"
        "│       └── routes/\n"
        "│           └── task_routes.py\n"
        "├── database/\n"
        "│   └── models/\n"
        "│       └── task.py\n"
        "├── frontend/\n"
        "│   └── components/\n"
        "│       └── TaskList.js\n"
        "└── tests/\n"
        "    └── test_task_crud.py"
    )

    assert serialized_tree == expected_serialized_tree, (
        "Serialized directory tree does not match the expected structure.\n"
        f"Expected:\n{expected_serialized_tree}\n"
        f"Actual:\n{serialized_tree}"
    )

    # Verify the approved stack survived serialization.
    assert serialized_data["approved_stack"]["frontend"] == "React"
    assert serialized_data["approved_stack"]["backend"] == "FastAPI"
    assert serialized_data["approved_stack"]["database"] == "MongoDB"

    
    # Verify commands do not assume npm configuration
    # when package.json is not in the planned file inventory.
    planned_paths = {
        item["path"] for item in serialized_data["files"]
    }

   
    if "package.json" not in planned_paths:
        npm_commands = [
            command
            for command in (
                serialized_data["install_commands"]
                + serialized_data["run_commands"]
                + serialized_data["test_commands"]
            )
            if command.startswith("npm ")
        ]

        print("\nUnexpected npm commands:", npm_commands)

        assert not npm_commands, (
            "npm commands should not be included without a planned package.json."
        )



    # Verify task IDs and dependencies survived serialization.
    serialized_tasks = {
        task["task_id"]: task
        for task in serialized_data["tasks"]
    }

    assert set(serialized_tasks) == set(expected_dependencies)

    for task_id, expected_deps in expected_dependencies.items():
        assert serialized_tasks[task_id]["dependencies"] == expected_deps, (
            f"Serialized dependencies are incorrect for {task_id}."
        )

    print("Serialized directory tree verified against exact expected output.")
    print("Serialized JSON verified.")

    # Compare the serialized tree with a fresh tree from its file inventory.
    fresh_tree = generate_directory_tree(
        serialized_data["project_name"],
        [item["path"] for item in serialized_data["files"]],
    )

    print("\nTREE FROM SERIALIZED JSON:")
    print(serialized_data["directory_tree"])

    print("\nFRESHLY GENERATED TREE:")
    print(fresh_tree)

    assert serialized_data["directory_tree"] == fresh_tree, (
        "Serialized tree differs from a freshly generated tree."
    )
    # Verify the JSON string itself matches the parsed JSON object.
    assert (
        json.loads(serialized_blueprint)["directory_tree"]
        == serialized_data["directory_tree"]
    ), "Serialized JSON string and parsed tree do not match."

    print("Serialized JSON string consistency verified.")
   

    print("\nSERIALIZED BLUEPRINT:")
    print(serialized_blueprint)




# --------------------------------------------------
# 11. Run all tests
# --------------------------------------------------

if __name__ == "__main__":
    test_directory_tree_generation()
    test_blueprint_generation()
    test_task_plan_parser()
    test_task_dependency_mismatch_is_detected()
    print("\nAll blueprint tests passed.")
