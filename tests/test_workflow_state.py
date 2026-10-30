import ast
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_function(source_path, function_name, namespace):
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    module = ast.Module(body=[function_node], type_ignores=[])
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace[function_name]


def test_task_planner_node_writes_plan_to_graph_state():
    plan = "# Implementation Plan\nvalidated"
    task_planner_node = load_function(
        PROJECT_ROOT / "main.py",
        "task_planner_node",
        {"create_task_plan": lambda **kwargs: plan},
    )

    update = task_planner_node(
        {"user_request": "request", "requirements": "req", "design": "design"}
    )

    assert update == {"task_plan": plan}


def test_resume_failure_syncs_consumed_interrupt_and_plan_state():
    session_state = {
        "thread_id": "same-thread",
        "task_plan": "",
        "waiting_for_human": True,
        "workflow_completed": False,
    }
    st = SimpleNamespace(session_state=session_state)

    class FakeGraph:
        def __init__(self):
            self.snapshot = SimpleNamespace(
                values={"user_request": "request"},
                next=("human_approval",),
                interrupts=(object(),),
            )
            self.calls = []

        def get_state(self, config):
            assert config["configurable"]["thread_id"] == "same-thread"
            return self.snapshot

        def invoke(self, command, config):
            self.calls.append((command, config))
            self.snapshot = SimpleNamespace(
                values={"user_request": "request", "task_plan": "validated plan"},
                next=("task_planner",),
                interrupts=(),
            )
            raise ValueError("downstream failure")

    graph = FakeGraph()

    class FakeCommand:
        def __init__(self, *, resume):
            self.resume = resume

    namespace = {
        "st": st,
        "graph": graph,
        "Command": FakeCommand,
        "sync_graph_state": load_function(
            PROJECT_ROOT / "ui.py",
            "sync_graph_state",
            {"st": st, "get_config": lambda: {"configurable": {"thread_id": "same-thread"}}, "graph": graph},
        ),
    }
    namespace["get_config"] = lambda: {"configurable": {"thread_id": "same-thread"}}
    resume_workflow = load_function(PROJECT_ROOT / "ui.py", "resume_workflow", namespace)

    import pytest

    with pytest.raises(ValueError, match="downstream failure"):
        resume_workflow("APPROVED", "")

    assert len(graph.calls) == 1
    assert graph.calls[0][0].resume == {"decision": "APPROVED", "feedback": ""}
    assert session_state["task_plan"] == "validated plan"
    assert session_state["waiting_for_human"] is False
    assert session_state["workflow_completed"] is False


def test_approval_resume_is_blocked_without_human_interrupt():
    session_state = {"thread_id": "same-thread"}
    st = SimpleNamespace(session_state=session_state)

    class FakeGraph:
        calls = 0

        def get_state(self, config):
            return SimpleNamespace(
                values={"user_request": "request"},
                next=("task_planner",),
                interrupts=(),
            )

        def invoke(self, command, config):
            self.calls += 1

    graph = FakeGraph()

    class FakeCommand:
        def __init__(self, *, resume):
            self.resume = resume

    namespace = {
        "st": st,
        "graph": graph,
        "Command": FakeCommand,
        "get_config": lambda: {"configurable": {"thread_id": "same-thread"}},
        "sync_graph_state": lambda: None,
    }
    resume_workflow = load_function(PROJECT_ROOT / "ui.py", "resume_workflow", namespace)

    import pytest

    with pytest.raises(ValueError, match="not waiting at a human approval"):
        resume_workflow("APPROVED", "")

    assert graph.calls == 0


def test_plan_rendering_is_gated_by_plan_presence_not_workflow_completion():
    tree = ast.parse((PROJECT_ROOT / "ui.py").read_text(encoding="utf-8"))
    plan_render_gate = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == 'st.session_state["task_plan"].strip()'
    )

    rendered_plan = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "markdown"
        and node.args
        and ast.unparse(node.args[0]) == 'st.session_state["task_plan"]'
        for node in ast.walk(plan_render_gate)
    )
    assert rendered_plan
