
import unittest

from workspace_tools import (
    list_workspace_files,
    read_workspace_file,
    write_workspace_file,
)


class WorkspaceToolsTests(unittest.TestCase):

    def setUp(self):
        self.test_file = "tool_test_sample.txt"

    def tearDown(self):
        # Clean up the file created by these tests.
        from workspace_tools import _resolve_workspace_path

        path = _resolve_workspace_path(self.test_file)
        if path.exists():
            path.unlink()

    def test_write_and_read_file(self):
        content = "Hello from the coding agent!"

        write_workspace_file(self.test_file, content)
        result = read_workspace_file(self.test_file)

        self.assertEqual(result, content)

    def test_list_files(self):
        write_workspace_file(self.test_file, "Test content")

        files = list_workspace_files()

        self.assertIn(self.test_file, files)

    def test_reject_parent_directory_traversal(self):
        with self.assertRaises(ValueError):
            write_workspace_file("../outside.txt", "Not allowed")

    def test_reject_absolute_paths(self):
        with self.assertRaises(ValueError):
            read_workspace_file("C:/outside.txt")

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            read_workspace_file("file_that_does_not_exist.txt")


if __name__ == "__main__":
    unittest.main()
