
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from coding_tools import coding_tools

load_dotenv()

MAX_TOOL_ROUNDS = 12
MAX_TEST_ATTEMPTS = 3

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

llm_with_tools = llm.bind_tools(coding_tools)
tool_map = {tool.name: tool for tool in coding_tools}


SYSTEM_PROMPT = """
You are a software coding agent.

Your task is to implement the user's requested changes inside
the designated workspace using the provided tools.

Workflow:
1. Inspect relevant files before changing them.
2. Implement the requested changes.
3. Run relevant tests using run_project_tests.
4. Examine the actual test output.
5. If tests fail, diagnose the cause and fix the code.
6. Rerun tests after a correction, provided the controller
   permits another test attempt.
7. Report the actual verification results.

Rules:
- Use only the provided tools.
- Never claim tests passed without successful test output.
- Do not claim you executed commands that you did not execute.
- Inspect existing files before overwriting them.
- Do not access files outside the designated workspace.
- Do not attempt to bypass execution restrictions.
- If a tool fails, explain the failure accurately.
- If the task cannot be completed safely, stop and explain why.

The controller enforces the test-attempt limit.
Do not repeatedly request tests after the controller
reports that the limit has been reached.
"""


def run_coding_agent(
    task: str,
    max_rounds: int = MAX_TOOL_ROUNDS,
    max_test_attempts: int = MAX_TEST_ATTEMPTS,
) -> str:
    if not isinstance(task, str) or not task.strip():
        return "Please provide a non-empty coding task."

    if (
        isinstance(max_rounds, bool)
        or not isinstance(max_rounds, int)
        or not 1 <= max_rounds <= MAX_TOOL_ROUNDS
    ):
        return f"max_rounds must be between 1 and {MAX_TOOL_ROUNDS}."

    if (
        isinstance(max_test_attempts, bool)
        or not isinstance(max_test_attempts, int)
        or not 1 <= max_test_attempts <= MAX_TEST_ATTEMPTS
    ):
        return (
            f"max_test_attempts must be between "
            f"1 and {MAX_TEST_ATTEMPTS}."
        )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=task),
    ]

    test_attempts = 0
    test_results = []
    tool_rounds = 0

    while tool_rounds < max_rounds:
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            report = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )

            if test_results:
                verified = any(
                    result["success"] for result in test_results
                )
                final_status = (
                    "At least one test run passed."
                    if verified
                    else "No test run passed."
                )
            else:
                final_status = "No tests were executed."

            return (
                f"{report}\n\n"
                f"--- Controller Verification ---\n"
                f"Test attempts: {test_attempts}/"
                f"{max_test_attempts}\n"
                f"Tool rounds: {tool_rounds}/{max_rounds}\n"
                f"Latest verification summary: {final_status}\n"
                f"Test results: {test_results}"
            )

        tool_rounds += 1
        limit_reached = False

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            if tool_name == "run_project_tests":
                if test_attempts >= max_test_attempts:
                    tool_result = (
                        "Test execution denied: maximum test attempts "
                        f"({max_test_attempts}) reached. Do not request "
                        "another test run. Provide the final report."
                    )
                    limit_reached = True

                else:
                    test_attempts += 1

                    try:
                        tool_result = tool_map[tool_name].invoke(
                            tool_args
                        )
                        result_text = str(tool_result)

                        # Record whether the runner explicitly reported
                        # success. A failed or malformed tool response
                        # is not treated as a passing test.
                        success = (
                            "Success: True" in result_text
                            and "Return code: 0" in result_text
                            and "Timed out: False" in result_text
                        )

                        test_results.append({
                            "attempt": test_attempts,
                            "success": success,
                            "output": result_text,
                        })

                    except Exception as exc:
                        tool_result = (
                            f"Test tool failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        test_results.append({
                            "attempt": test_attempts,
                            "success": False,
                            "output": str(tool_result),
                        })

            else:
                selected_tool = tool_map.get(tool_name)

                if selected_tool is None:
                    tool_result = (
                        f"Error: Unknown tool '{tool_name}'."
                    )
                else:
                    try:
                        tool_result = selected_tool.invoke(tool_args)
                    except Exception as exc:
                        tool_result = (
                            f"Tool execution failed: "
                            f"{type(exc).__name__}: {exc}"
                        )

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call_id,
                )
            )

        if limit_reached:
            messages.append(
                HumanMessage(
                    content=(
                        "The test-attempt limit has been reached. "
                        "Do not call any more tools. Provide a final "
                        "report describing the last test result and "
                        "any remaining issues."
                    )
                )
            )

    return (
        "Stopped because the tool-call round limit was reached. "
        f"Test attempts: {test_attempts}/{max_test_attempts}. "
        "Inspect the workspace and test results before deciding "
        "whether the task is complete."
    )


if __name__ == "__main__":
    user_task = input("Enter your coding task: ").strip()

    if not user_task:
        print("Please provide a coding task.")
    else:
        try:
            result = run_coding_agent(user_task)
            print("\n--- Coding Agent Report ---\n")
            print(result)
        except Exception as exc:
            print(
                f"Coding agent failed: "
                f"{type(exc).__name__}: {exc}"
            )
