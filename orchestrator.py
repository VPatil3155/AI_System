from __future__ import annotations

from typing import Any


SUPPORTED_WORKERS = {
    "frontend",
    "backend",
    "database",
    "testing",
    "shared",
    "configuration",
}


def _task_value(task: Any, field: str, default: Any = None) -> Any:
    """Read a field from either a Pydantic model or a dictionary."""
    if isinstance(task, dict):
        return task.get(field, default)

    return getattr(task, field, default)


def initialize_task_registry(blueprint: Any) -> dict[str, dict]:
    """
    Create the initial execution registry from a validated ProjectBlueprint.

    All tasks start as pending. This function does not execute code.
    """
    tasks = _task_value(blueprint, "tasks", [])

    if not tasks:
        raise ValueError("Cannot initialize the orchestrator: blueprint has no tasks.")

    registry: dict[str, dict] = {}

    for task in tasks:
        task_id = _task_value(task, "task_id")
        worker_type = _task_value(task, "worker_type")
        dependencies = list(_task_value(task, "dependencies", []) or [])
        files = list(_task_value(task, "files", []) or [])

        if not task_id:
            raise ValueError("Every task must have a non-empty task_id.")

        if task_id in registry:
            raise ValueError(f"Duplicate task ID: {task_id}")

        if worker_type not in SUPPORTED_WORKERS:
            raise ValueError(
                f"Task {task_id} has unsupported worker type: {worker_type}"
            )

        registry[task_id] = {
            "task_id": task_id,
            "title": _task_value(task, "title", task_id),
            "worker_type": worker_type,
            "objective": _task_value(task, "objective", ""),
            "files": files,
            "dependencies": dependencies,
            "acceptance_criteria": list(
                _task_value(task, "acceptance_criteria", []) or []
            ),
            "status": "pending",
            "attempts": 0,
            "error": None,
        }

    # Reject dependencies that reference unknown tasks.
    for task_id, task in registry.items():
        missing = set(task["dependencies"]) - set(registry)

        if missing:
            raise ValueError(
                f"Task {task_id} has unknown dependencies: {sorted(missing)}"
            )

        if task_id in task["dependencies"]:
            raise ValueError(f"Task {task_id} cannot depend on itself.")

    _validate_acyclic_dependencies(registry)

    return registry


def _validate_acyclic_dependencies(registry: dict[str, dict]) -> None:
    """Reject dependency cycles before execution begins."""
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise ValueError(
                f"Dependency cycle detected involving task {task_id}."
            )

        if task_id in visited:
            return

        visiting.add(task_id)

        for dependency in registry[task_id]["dependencies"]:
            visit(dependency)

        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in registry:
        visit(task_id)


def get_ready_tasks(registry: dict[str, dict]) -> list[dict]:
    """
    Return pending tasks whose dependencies are all completed.

    The result preserves the original task order.
    """
    ready = []

    for task in registry.values():
        if task["status"] != "pending":
            continue

        dependencies = task["dependencies"]

        if all(
            registry[dependency]["status"] == "completed"
            for dependency in dependencies
        ):
            ready.append(task)

    return ready


def select_next_task(registry: dict[str, dict]) -> dict | None:
    """
    Select the first ready task for sequential execution.

    Returns None if there are no ready tasks.
    """
    ready_tasks = get_ready_tasks(registry)

    if not ready_tasks:
        return None

    return ready_tasks[0]


def update_task_status(
    registry: dict[str, dict],
    task_id: str,
    status: str,
    error: str | None = None,
) -> None:
    """Update task status after a worker or verifier reports a result."""
    allowed_statuses = {
        "pending",
        "running",
        "completed",
        "failed",
        "blocked",
    }

    if task_id not in registry:
        raise KeyError(f"Unknown task ID: {task_id}")

    if status not in allowed_statuses:
        raise ValueError(f"Unsupported task status: {status}")

    task = registry[task_id]

    if status == "running":
        if task["status"] != "pending":
            raise ValueError(
                f"Cannot start task {task_id} from status {task['status']}."
            )

        # Dependencies must be complete before execution.
        incomplete = [
            dep
            for dep in task["dependencies"]
            if registry[dep]["status"] != "completed"
        ]

        if incomplete:
            raise ValueError(
                f"Cannot start {task_id}; dependencies are incomplete: "
                f"{incomplete}"
            )

        task["attempts"] += 1

    task["status"] = status
    task["error"] = error


def get_execution_summary(registry: dict[str, dict]) -> dict:
    """Return task counts and the remaining ready work."""
    counts = {
        "pending": 0,
        "running": 0,
        "completed": 0,
        "failed": 0,
        "blocked": 0,
    }

    for task in registry.values():
        counts[task["status"]] += 1

    ready_tasks = get_ready_tasks(registry)

    return {
        "total_tasks": len(registry),
        "counts": counts,
        "ready_task_ids": [task["task_id"] for task in ready_tasks],
        "all_completed": counts["completed"] == len(registry),
        "has_failures": counts["failed"] > 0,
    }