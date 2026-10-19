import pytest

from orchestrator import (
    initialize_task_registry,
    get_ready_tasks,
    select_next_task,
    update_task_status,
    get_execution_summary,
)


@pytest.fixture
def blueprint():
    return {
        "tasks": [
            {
                "task_id": "DB-001",
                "title": "Create database schema",
                "worker_type": "database",
                "objective": "Define the database schema.",
                "files": ["database/schema.sql"],
                "dependencies": [],
                "acceptance_criteria": ["Schema is defined."],
            },
            {
                "task_id": "BE-001",
                "title": "Create API",
                "worker_type": "backend",
                "objective": "Implement the API.",
                "files": ["backend/api.py"],
                "dependencies": ["DB-001"],
                "acceptance_criteria": ["API is implemented."],
            },
            {
                "task_id": "TEST-001",
                "title": "Test API",
                "worker_type": "testing",
                "objective": "Test the API.",
                "files": ["tests/test_api.py"],
                "dependencies": ["BE-001"],
                "acceptance_criteria": ["Tests are defined."],
            },
        ]
    }


def test_only_dependency_free_tasks_are_ready(blueprint):
    registry = initialize_task_registry(blueprint)

    ready = get_ready_tasks(registry)

    assert [task["task_id"] for task in ready] == ["DB-001"]


def test_next_task_selected_in_blueprint_order(blueprint):
    registry = initialize_task_registry(blueprint)

    task = select_next_task(registry)

    assert task["task_id"] == "DB-001"


def test_completed_dependency_unlocks_next_task(blueprint):
    registry = initialize_task_registry(blueprint)

    update_task_status(registry, "DB-001", "running")
    update_task_status(registry, "DB-001", "completed")

    ready = get_ready_tasks(registry)

    assert [task["task_id"] for task in ready] == ["BE-001"]


def test_cannot_start_task_before_dependencies_complete(blueprint):
    registry = initialize_task_registry(blueprint)

    with pytest.raises(ValueError, match="dependencies are incomplete"):
        update_task_status(registry, "BE-001", "running")


def test_summary_tracks_completed_tasks(blueprint):
    registry = initialize_task_registry(blueprint)

    update_task_status(registry, "DB-001", "running")
    update_task_status(registry, "DB-001", "completed")

    summary = get_execution_summary(registry)

    assert summary["total_tasks"] == 3
    assert summary["counts"]["completed"] == 1
    assert summary["ready_task_ids"] == ["BE-001"]


def test_dependency_cycle_is_rejected():
    invalid_blueprint = {
        "tasks": [
            {
                "task_id": "A",
                "worker_type": "backend",
                "dependencies": ["B"],
            },
            {
                "task_id": "B",
                "worker_type": "backend",
                "dependencies": ["A"],
            },
        ]
    }

    with pytest.raises(ValueError, match="Dependency cycle"):
        initialize_task_registry(invalid_blueprint)