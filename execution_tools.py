
import subprocess
import sys
from pathlib import Path

from workspace_tools import WORKSPACE_DIR


MAX_TIMEOUT_SECONDS = 30
MAX_OUTPUT_CHARS = 12000


def run_workspace_tests(
    test_file: str = "test_calculator.py",
    timeout_seconds: int = 20,
) -> dict:
    """
    Run a single approved unittest file inside workspace/.

    Only test files with names beginning with test_ and ending
    with .py are accepted. Arbitrary shell commands are not allowed.
    """

    # 1. Validate the requested test filename.
    if not isinstance(test_file, str) or not test_file.strip():
        raise ValueError("A test filename is required.")

    requested_path = Path(test_file)

    if (
        requested_path.is_absolute()
        or len(requested_path.parts) != 1
        or requested_path.name != test_file
        or not test_file.startswith("test_")
        or not test_file.endswith(".py")
    ):
        raise ValueError(
            "Provide a filename like test_calculator.py, "
            "not an absolute path or a path containing directories."
        )

    # 2. Validate the timeout.
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or not 1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS
    ):
        raise ValueError(
            f"Timeout must be between 1 and {MAX_TIMEOUT_SECONDS} seconds."
        )

    # 3. Resolve and validate the test file.
    workspace = WORKSPACE_DIR.resolve()
    test_path = (workspace / test_file).resolve()

    if not test_path.is_relative_to(workspace):
        raise ValueError("The test file must remain inside workspace/.")

    if not test_path.is_file():
        raise FileNotFoundError(
            f"Test file not found inside workspace: {test_file}"
        )

    # 4. Run a fixed command without invoking a shell.
    command = [
        sys.executable,
        "-m",
        "unittest",
        "-v",
        test_file,
    ]

    try:
        result = subprocess.run(
            command,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=False,
            check=False,
        )

        output = (
            (result.stdout or "")
            + "\n"
            + (result.stderr or "")
        ).strip()

        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "timed_out": False,
            "output": output[:MAX_OUTPUT_CHARS],
            "output_truncated": len(output) > MAX_OUTPUT_CHARS,
        }

    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""

        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")

        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")

        output = (stdout + "\n" + stderr).strip()

        return {
            "success": False,
            "returncode": None,
            "timed_out": True,
            "output": (
                output[:MAX_OUTPUT_CHARS]
                + "\nTest execution exceeded the time limit."
            ),
            "output_truncated": len(output) > MAX_OUTPUT_CHARS,
        }
