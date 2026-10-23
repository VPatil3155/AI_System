
from types import SimpleNamespace

import pytest

from orchestrator import (
    initialize_task_registry,
    get_ready_tasks,
    select_next_task,
    update_task_status,
    get_execution_summary,
)


def make_task(
    task_id,
    worker_type="backend",
    dependencies=None,
):
    return {
        "task_id": task_id,
        "title": f"Task {task_id}",
        "worker_type": worker_type,
        "objective": f"Implement {task_id}",
        "files": [f"backend/{task_id.lower()}.py"],
        "dependencies": dependencies or [],
        "acceptance_criteria": ["Expected behavior is verified"],
    }


def make_blueprint(*tasks):
    return SimpleNamespace(
        tasks=list(tasks),
        project_name="Test Project",
    )


def test_initialize_registry_sets_tasks_to_pending():
    blueprint = make_blueprint(
        make_task("DB-001", "database"),
        make_task("BE-001", dependencies=["DB-001"]),
    )

    registry = initialize_task_registry(blueprint)

    assert list(registry) == ["DB-001", "BE-001"]
    assert registry["DB-001"]["status"] == "pending"
    assert registry["BE-001"]["status"] == "pending"
    assert registry["BE-001"]["attempts"] == 0


def test_only_tasks_without_incomplete_dependencies_are_ready():
    blueprint = make_blueprint(
        make_task("DB-001", "database"),
        make_task("BE-001", dependencies=["DB-001"]),
        make_task("FE-001", "frontend"),
    )

    registry = initialize_task_registry(blueprint)

    assert [
        task["task_id"] for task in get_ready_tasks(registry)
    ] == ["DB-001", "FE-001"]


def test_completing_dependency_unlocks_dependent_task():
    blueprint = make_blueprint(
        make_task("DB-001", "database"),
        make_task("BE-001", dependencies=["DB-001"]),
    )

    registry = initialize_task_registry(blueprint)

    update_task_status(registry, "DB-001", "running")
    update_task_status(registry, "DB-001", "completed")

    selected = select_next_task(registry)

    assert selected is not None
    assert selected["task_id"] == "BE-001"


def test_cannot_start_task_before_dependencies_complete():
    blueprint = make_blueprint(
        make_task("DB-001", "database"),
        make_task("BE-001", dependencies=["DB-001"]),
    )

    registry = initialize_task_registry(blueprint)

    with pytest.raises(ValueError, match="dependencies are incomplete"):
        update_task_status(registry, "BE-001", "running")


def test_unknown_dependency_is_rejected():
    blueprint = make_blueprint(
        make_task("BE-001", dependencies=["DB-999"]),
    )

    with pytest.raises(ValueError, match="unknown dependencies"):
        initialize_task_registry(blueprint)


def test_dependency_cycle_is_rejected():
    blueprint = make_blueprint(
        make_task("BE-001", dependencies=["DB-001"]),
        make_task("DB-001", "database", dependencies=["BE-001"]),
    )

    with pytest.raises(ValueError, match="Dependency cycle"):
        initialize_task_registry(blueprint)


def test_task_cannot_be_started_twice():
    registry = initialize_task_registry(
        make_blueprint(make_task("BE-001"))
    )

    update_task_status(registry, "BE-001", "running")

    with pytest.raises(ValueError, match="Cannot start task"):
        update_task_status(registry, "BE-001", "running")


def test_execution_summary_reports_completion():
    registry = initialize_task_registry(
        make_blueprint(make_task("BE-001"))
    )

    update_task_status(registry, "BE-001", "running")
    update_task_status(registry, "BE-001", "completed")

    summary = get_execution_summary(registry)

    assert summary["total_tasks"] == 1
    assert summary["counts"]["completed"] == 1
    assert summary["all_completed"] is True
    assert summary["has_failures"] is False


def test_empty_blueprint_is_rejected():
    with pytest.raises(ValueError, match="blueprint has no tasks"):
        initialize_task_registry(make_blueprint())


def test_dictionary_and_object_task_inputs_are_supported():
    task = SimpleNamespace(
        task_id="BE-001",
        worker_type="backend",
        dependencies=[],
        files=["backend/app.py"],
        acceptance_criteria=["Application starts"],
        objective="Build backend",
        title="Backend",
    )

    registry = initialize_task_registry(make_blueprint(task))

    assert registry["BE-001"]["objective"] == "Build backend"
    assert registry["BE-001"]["files"] == ["backend/app.py"]
