from typing import Any
from coding_agent import run_coding_agent


def worker_node(state: dict[str, Any]) -> dict:
    task = state.get("task")

    if not isinstance(task, dict):
        raise ValueError("Worker requires a task dictionary.")

    task_id = task["task_id"]

    instruction = f"""
Implement the following software task.

Task ID: {task_id}
Worker type: {task["worker_type"]}
Title: {task.get("title", task_id)}
Objective: {task["objective"]}
Files: {task.get("files", [])}
Acceptance criteria: {task.get("acceptance_criteria", [])}

Inspect existing files before modifying them.
Follow the task scope and acceptance criteria.
Report what you changed and the actual test results.
"""

    try:
        report = run_coding_agent(instruction)

        tests_passed = (
            "Latest verification summary: At least one test run passed."
            in report
        )

        return {
            "worker_results": [
                {
                    "task_id": task_id,
                    "worker_type": task["worker_type"],
                    "status": "completed" if tests_passed else "failed",
                    "message": report,
                    "error": None if tests_passed else (
                        "Coding agent did not confirm a passing test run."
                    ),
                }
            ]
        }

    except Exception as exc:
        return {
            "worker_results": [
                {
                    "task_id": task_id,
                    "worker_type": task["worker_type"],
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            ]
        }