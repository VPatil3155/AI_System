from langchain.tools import tool

from workspace_tools import (
    list_workspace_files,
    read_workspace_file,
    write_workspace_file,
)
from execution_tools import run_workspace_tests


@tool
def list_project_files() -> str:
    """List the files available inside the designated software workspace."""
    files = list_workspace_files()

    if not files:
        return "The workspace is empty."

    return "\n".join(files)


@tool
def read_project_file(relative_path: str) -> str:
    """Read a UTF-8 text file from the workspace using its relative path."""
    return read_workspace_file(relative_path)


@tool
def write_project_file(relative_path: str, content: str) -> str:
    """Create or overwrite a UTF-8 text file inside the workspace.

    Use a relative path. Never use absolute paths or paths outside the
    workspace. Existing files will be overwritten, so inspect them first.
    """
    return write_workspace_file(relative_path, content)


@tool
def run_project_tests(test_file: str = "test_calculator.py") -> str:
    """Run a Python unittest file from the designated workspace.

    Use a test filename such as test_calculator.py.
    Returns the actual test output, exit code, and execution status.
    """
    result = run_workspace_tests(test_file)

    return (
        f"Success: {result['success']}\n"
        f"Return code: {result['returncode']}\n"
        f"Timed out: {result['timed_out']}\n"
        f"Output truncated: {result['output_truncated']}\n"
        f"Test output:\n{result['output']}"
    )


coding_tools = [
    list_project_files,
    read_project_file,
    write_project_file,
    run_project_tests,
]