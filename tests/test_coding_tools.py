
import unittest

from coding_tools import (
    list_project_files,
    read_project_file,
    write_project_file,
)


class CodingToolsTests(unittest.TestCase):

    def setUp(self):
        self.filename = "langchain_tool_test.txt"

    def tearDown(self):
        from workspace_tools import _resolve_workspace_path

        path = _resolve_workspace_path(self.filename)

        if path.exists():
            path.unlink()

    def test_write_tool(self):
        result = write_project_file.invoke({
            "relative_path": self.filename,
            "content": "Created through LangChain",
        })

        self.assertIn("Successfully wrote", result)

    def test_read_tool(self):
        write_project_file.invoke({
            "relative_path": self.filename,
            "content": "Hello from the tool",
        })

        result = read_project_file.invoke({
            "relative_path": self.filename,
        })

        self.assertEqual(result, "Hello from the tool")

    def test_list_tool(self):
        write_project_file.invoke({
            "relative_path": self.filename,
            "content": "Test",
        })

        result = list_project_files.invoke({})

        self.assertIn(self.filename, result)


if __name__ == "__main__":
    unittest.main()
