
from pathlib import Path
import re

from blueprint_agent import ProjectBlueprint
from workspace_tools import WORKSPACE_DIR


def scaffold_project(
    blueprint: ProjectBlueprint,
    overwrite: bool = False,
) -> dict:
    """
    Create a project structure from a validated ProjectBlueprint.

    Files are created empty. Existing files are preserved unless
    overwrite=True is explicitly requested.
    """

    if not isinstance(blueprint, ProjectBlueprint):
        raise TypeError("blueprint must be a ProjectBlueprint instance.")

    project_name = blueprint.project_name.strip()

    # Validate the project directory name.
    if (
        not project_name
        or project_name in {".", ".."}
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", project_name)
    ):
        raise ValueError(f"Invalid project name: {project_name!r}")

    workspace = WORKSPACE_DIR.resolve()
    project_root = (workspace / project_name).resolve()

    if not project_root.is_relative_to(workspace):
        raise ValueError("Project directory must remain inside workspace.")

    created_files = []
    skipped_files = []

    # Validate all file paths before creating anything.
    validated_paths = []

    for blueprint_file in blueprint.files:
        relative_path = blueprint_file.path.replace("\\", "/")
        parts = relative_path.split("/")

        if (
            not relative_path
            or relative_path.startswith("/")
            or any(part in {"", ".", ".."} for part in parts)
            or (len(parts[0]) >= 2 and parts[0][1] == ":")
        ):
            raise ValueError(f"Unsafe blueprint file path: {relative_path!r}")

        destination = (project_root / Path(*parts)).resolve()

        if not destination.is_relative_to(project_root):
            raise ValueError(f"File escapes project directory: {relative_path!r}")

        if destination.name.lower() in {
            ".env",
            ".env.local",
            ".env.production",
            "credentials.json",
        }:
            raise ValueError(f"Blocked filename: {destination.name}")

        validated_paths.append((relative_path, destination))

    # Avoid accidental overwrites unless explicitly enabled.
    if not overwrite:
        existing = [
            relative_path
            for relative_path, destination in validated_paths
            if destination.exists()
        ]
        if existing:
            raise FileExistsError(
                "These project files already exist: "
                + ", ".join(existing)
                + ". Use overwrite=True only if replacement is intended."
            )

    # Create directories and empty files.
    project_root.mkdir(parents=True, exist_ok=True)

    for relative_path, destination in validated_paths:
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists() and destination.is_dir():
            raise IsADirectoryError(
                f"Expected a file but found a directory: {relative_path}"
            )

        if destination.exists() and not overwrite:
            skipped_files.append(relative_path)
            continue

        destination.write_text("", encoding="utf-8")
        created_files.append(relative_path)

    return {
        "success": True,
        "project_name": project_name,
        "project_path": str(project_root),
        "created_files": created_files,
        "skipped_files": skipped_files,
        "total_files": len(validated_paths),
    }
