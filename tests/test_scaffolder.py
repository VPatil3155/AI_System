
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from pydantic import ValidationError
from blueprint_agent import (
    ProjectBlueprint,
    TechnologyStack,
    BlueprintFile,
)
from scaffolder import scaffold_project


def make_blueprint(project_name="test-project", paths=None):
    paths = paths or ["src/main.py", "tests/test_main.py"]

    return ProjectBlueprint(
        project_name=project_name,
        project_summary="Scaffolder test project",
        approved_stack=TechnologyStack(
            frontend="Not applicable",
            backend="Python",
            database="Not applicable",
        ),
        architecture_style="Not specified",
        directory_tree="",
        files=[
            BlueprintFile(
                path=path,
                purpose="Test file",
                owner="shared",
            )
            for path in paths
        ],
        tasks=[],
    )


class ScaffolderTests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)

        self.workspace_patch = patch(
            "scaffolder.WORKSPACE_DIR",
            self.workspace,
        )
        self.workspace_patch.start()

    def tearDown(self):
        self.workspace_patch.stop()
        self.temp_dir.cleanup()

    def test_creates_project_files(self):
        blueprint = make_blueprint()

        result = scaffold_project(blueprint)

        self.assertTrue(result["success"])
        self.assertTrue(
            (self.workspace / "test-project" / "src" / "main.py").is_file()
        )
        self.assertTrue(
            (self.workspace / "test-project" / "tests" / "test_main.py").is_file()
        )
        self.assertEqual(len(result["created_files"]), 2)

    def test_rejects_unsafe_path(self):
        with self.assertRaises(ValidationError):
            make_blueprint(paths=["../outside.py"])

        self.assertFalse((self.workspace.parent / "outside.py").exists())

    def test_does_not_overwrite_existing_file(self):
        blueprint = make_blueprint(paths=["src/main.py"])
        existing_file = (
            self.workspace / "test-project" / "src" / "main.py"
        )
        existing_file.parent.mkdir(parents=True)
        existing_file.write_text("keep this content", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            scaffold_project(blueprint)

        self.assertEqual(
            existing_file.read_text(encoding="utf-8"),
            "keep this content",
        )

    def test_rejects_invalid_project_name(self):
        blueprint = make_blueprint(project_name="../outside")

        with self.assertRaises(ValueError):
            scaffold_project(blueprint)


if __name__ == "__main__":
    unittest.main()
