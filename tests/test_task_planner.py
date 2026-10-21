import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class UnavailableLLMClient:
    def __init__(self, **kwargs):
        pass

    def invoke(self, *args, **kwargs):
        raise AssertionError("LLM clients are not used by validation tests.")


for module_name, class_name in (
    ("langchain_ollama", "ChatOllama"),
    ("langchain_groq", "ChatGroq"),
):
    if importlib.util.find_spec(module_name) is None:
        llm_module = ModuleType(module_name)
        setattr(llm_module, class_name, UnavailableLLMClient)
        sys.modules[module_name] = llm_module

import task_planner


def make_task_block(fields: str) -> str:
    return f"### DB-001: Define the task data model\n{fields}\n"


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakePlanner:
    def __init__(self, plans):
        self.plans = plans
        self.calls = 0

    def invoke(self, prompt):
        plan = self.plans[self.calls]
        self.calls += 1
        return FakeResponse(plan)


def assert_plan_rejected_after_one_repair(monkeypatch, task):
    fake_planner = FakePlanner([task, task])
    monkeypatch.setattr(task_planner, "planner_llm", fake_planner)

    with pytest.raises(ValueError, match="after the repair attempt"):
        task_planner.create_task_plan("request", "requirements", "design")

    assert fake_planner.calls == 2


def test_valid_task_with_bold_field_labels(monkeypatch):
    task = make_task_block(
        """- **Task ID:** DB-001
- **Worker type:** database
- **Objective:** Define the persistent task model.
- **Files to create or modify:**
  - main.py
- **Scope and implementation details:** Define the task fields.
- **Dependencies (task IDs):** None
- **Acceptance criteria:**
  - The model contains the required fields."""
    )

    fake_planner = FakePlanner([task])
    monkeypatch.setattr(task_planner, "planner_llm", fake_planner)
    assert task_planner.create_task_plan("request", "requirements", "design") == task.strip()
    assert fake_planner.calls == 1


def test_valid_task_with_plain_field_labels(monkeypatch):
    task = make_task_block(
        """- Task ID: DB-001
- Worker type: database
- Objective: Define the persistent task model.
- Files to create or modify:
  - database/models/task.py
- Scope and implementation details: Define the task fields.
- Dependencies (task IDs): None
- Acceptance criteria:
  - The model contains the required fields."""
    )

    fake_planner = FakePlanner([task])
    monkeypatch.setattr(task_planner, "planner_llm", fake_planner)
    assert task_planner.create_task_plan("request", "requirements", "design") == task.strip()
    assert fake_planner.calls == 1


def test_task_missing_task_id_is_rejected(monkeypatch):
    task = make_task_block(
        """- Worker type: database
- Objective: Define the persistent task model.
- Files to create or modify:
  - database/models/task.py
- Dependencies (task IDs): None
- Acceptance criteria:
  - The model contains the required fields."""
    )

    assert_plan_rejected_after_one_repair(monkeypatch, task)


def test_task_with_mismatched_task_id_is_rejected(monkeypatch):
    task = make_task_block(
        """- **Task ID:** BE-001
- **Worker type:** database
- **Objective:** Define the persistent task model.
- **Files to create or modify:**
  - database/models/task.py
- **Dependencies (task IDs):** None
- **Acceptance criteria:**
  - The model contains the required fields."""
    )

    assert_plan_rejected_after_one_repair(monkeypatch, task)


def test_task_missing_scope_is_rejected(monkeypatch):
    task = make_task_block(
        """- Task ID: DB-001
- Worker type: database
- Objective: Define the persistent task model.
- Files to create or modify:
  - database/models/task.py
- Dependencies (task IDs): None
- Acceptance criteria:
  - The model contains the required fields."""
    )

    assert_plan_rejected_after_one_repair(monkeypatch, task)


def test_task_missing_acceptance_criteria_bullets_is_rejected(monkeypatch):
    task = make_task_block(
        """- Task ID: DB-001
- Worker type: database
- Objective: Define the persistent task model.
- Files to create or modify:
  - database/models/task.py
- Dependencies (task IDs): None
- Acceptance criteria:"""
    ) + "## 7. Execution Order\n- DB-001\n"

    assert_plan_rejected_after_one_repair(monkeypatch, task)


def test_task_missing_repository_relative_file_path_is_rejected(monkeypatch):
    task = make_task_block(
        """- Task ID: DB-001
- Worker type: database
- Objective: Define the persistent task model.
- Files to create or modify:
  - To be determined
- Dependencies (task IDs): None
- Acceptance criteria:
  - The model contains the required fields."""
    )

    assert_plan_rejected_after_one_repair(monkeypatch, task)


@pytest.mark.parametrize("file_path", ["/etc/app.py", "../outside.py", "C:\\app\\main.py"])
def test_task_with_non_relative_file_path_is_rejected(monkeypatch, file_path):
    task = make_task_block(
        f"""- Task ID: DB-001
- Worker type: database
- Objective: Define the persistent task model.
- Files to create or modify:
  - {file_path}
- Scope and implementation details: Define the task fields.
- Dependencies (task IDs): None
- Acceptance criteria:
  - The model contains the required fields."""
    )

    assert_plan_rejected_after_one_repair(monkeypatch, task)
