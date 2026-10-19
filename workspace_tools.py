
from pathlib import Path

# Only this directory is accessible through these tools.
BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = (BASE_DIR / "workspace").resolve()

# Directories that should not appear in file listings.
IGNORED_DIRECTORIES = {
    ".git",
    "__pycache__",
    ".venv",
    "node_modules",
}

# Files that must never be read or overwritten through these tools.
BLOCKED_FILENAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
}


def _resolve_workspace_path(relative_path: str) -> Path:
    """Resolve a relative path and ensure it stays inside workspace/."""

    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("A non-empty relative path is required.")

    requested_path = Path(relative_path)

    if requested_path.is_absolute():
        raise ValueError("Absolute paths are not allowed.")

    if any(part in {".", ".."} for part in requested_path.parts):
        raise ValueError("Use a normalized relative path without '.' or '..'.")

    resolved_path = (WORKSPACE_DIR / requested_path).resolve()

    if not resolved_path.is_relative_to(WORKSPACE_DIR):
        raise ValueError("Access outside the workspace is not allowed.")

    if resolved_path.name.lower() in BLOCKED_FILENAMES:
        raise ValueError("Access to this filename is blocked.")

    return resolved_path


def list_workspace_files() -> list[str]:
    """Return the relative paths of files inside workspace/."""

    files = []

    for path in WORKSPACE_DIR.rglob("*"):
        if any(part in IGNORED_DIRECTORIES for part in path.parts):
            continue

        # Resolve paths to avoid following links outside the workspace.
        try:
            resolved_path = path.resolve()
            if not resolved_path.is_relative_to(WORKSPACE_DIR):
                continue
        except (OSError, RuntimeError):
            continue

        if path.is_file() and path.name.lower() not in BLOCKED_FILENAMES:
            files.append(path.relative_to(WORKSPACE_DIR).as_posix())

    return sorted(files)


def read_workspace_file(relative_path: str) -> str:
    """Read a UTF-8 text file from workspace/."""

    path = _resolve_workspace_path(relative_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {relative_path}")

    if not path.is_file():
        raise ValueError("The requested path is not a file.")

    return path.read_text(encoding="utf-8")


def write_workspace_file(relative_path: str, content: str) -> str:
    """Create or overwrite a UTF-8 text file inside workspace/."""

    if not isinstance(content, str):
        raise TypeError("File content must be a string.")

    path = _resolve_workspace_path(relative_path)

    if path.exists() and not path.is_file():
        raise ValueError("The requested path is not a file.")

    # Create parent directories only after validating the destination.
    path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve again in case a directory or symbolic link changed.
    path = _resolve_workspace_path(relative_path)

    if path.exists() and not path.is_file():
        raise ValueError("The requested path is not a file.")

    path.write_text(content, encoding="utf-8")

    return f"Successfully wrote {relative_path}"
