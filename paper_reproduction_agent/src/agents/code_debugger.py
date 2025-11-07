"""Code Debugger Agent - Fixes broken or failing implementations."""

from typing import TypedDict
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from ..tools.code_execution_tools import code_execution_tools
from ..utils.llm_factory import create_llm
from ..utils.message_utils import normalize_message_content


class CodeDebuggerState(TypedDict):
    """State for Code Debugger Agent."""
    code_path: str
    error_messages: list
    fixes_applied: list
    code_fixed: bool
    fix_report: str


class CodeDebuggerAgent:
    """Agent responsible for debugging and fixing code."""

    def __init__(self, llm=None):
        """Initialize the Code Debugger Agent."""
        self.llm = llm or create_llm(temperature=0.1)

        self.system_prompt = """You are an expert debugging engineer specialized in fixing research code.

Your responsibilities:
1. Analyze error messages and stack traces
2. Identify root causes of failures
3. Fix bugs and errors systematically
4. Verify fixes work correctly
5. Document all changes made

Debugging approach:
- Read and understand error messages carefully
- Check syntax errors first
- Verify dependencies are installed correctly
- Fix import errors and missing modules
- Address runtime errors
- Fix logical errors causing incorrect results
- Test fixes to ensure they work

Common issues in research code:
- Missing dependencies
- Incorrect file paths
- API changes in libraries
- Tensor/array shape mismatches
- Missing data files
- Configuration errors
- Version incompatibilities

When fixing:
- Make minimal changes needed to fix the issue
- Preserve the original algorithm logic
- Add comments explaining fixes
- Test after each fix
- Document what was fixed and why

Use the available tools to diagnose and fix issues effectively."""

        self.agent = create_react_agent(
            self.llm,
            tools=code_execution_tools,
            prompt=self.system_prompt
        )

    def debug_and_fix(self, code_path: str, error_info: dict) -> dict:
        """
        Debug and fix code based on error information.

        Args:
            code_path: Path to code with errors
            error_info: Error messages and context

        Returns:
            Fix results
        """
        task = f"""Debug and fix the code at: {code_path}

Error information:
{error_info}

Debugging process:
1. Analyze the error messages
2. Check code syntax
3. Identify the root cause
4. Apply fixes
5. Verify the fixes work
6. Run the code again to confirm

Provide:
- Root cause analysis
- Fixes applied
- Verification results
- Updated code status

IMPORTANT: After completing your analysis and any fixes, you MUST provide a final answer and stop. Do not keep running tools indefinitely."""

        messages = [HumanMessage(content=task)]

        # Enable debug logging
        print("🔍 DEBUG: Invoking code debugger agent...")
        print(f"🔍 DEBUG: Code path: {code_path}")
        print(f"🔍 DEBUG: Error info: {error_info}")

        result = self.agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 50}
        )

        # Log the result
        print(f"🔍 DEBUG: Agent completed with {len(result.get('messages', []))} messages")
        for i, msg in enumerate(result.get('messages', [])):
            if hasattr(msg, 'content'):
                print(f"🔍 DEBUG: Message {i}: {msg.content[:200]}...")

        return self._parse_debug_result(result)

    def fix_dependency_issues(self, code_path: str, dependency_errors: list) -> dict:
        """
        Fix dependency-related errors.

        Args:
            code_path: Path to code
            dependency_errors: List of dependency errors

        Returns:
            Fix results
        """
        task = f"""Fix dependency issues in {code_path}:

Errors:
{dependency_errors}

Steps:
1. Identify missing packages
2. Check version conflicts
3. Update requirements.txt if needed
4. Install correct versions
5. Verify imports work

Document all dependency fixes."""

        messages = [HumanMessage(content=task)]
        result = self.agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 50}
        )

        return {
            "fixes": normalize_message_content(result.get("messages", [])[-1].content) if result.get("messages") else "",
        }

    def fix_runtime_errors(self, code_path: str, runtime_error: str, traceback: str) -> dict:
        """
        Fix runtime errors in code.

        Args:
            code_path: Path to code
            runtime_error: Runtime error message
            traceback: Full traceback

        Returns:
            Fix results
        """
        task = f"""Fix runtime error in {code_path}:

Error: {runtime_error}

Traceback:
{traceback}

Analysis:
1. Identify the exact line causing the error
2. Understand why it's failing
3. Determine the correct fix
4. Apply the fix
5. Test the fix

Provide detailed fix explanation and updated code."""

        messages = [HumanMessage(content=task)]
        result = self.agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 50}
        )

        return {
            "error": runtime_error,
            "fix": normalize_message_content(result.get("messages", [])[-1].content) if result.get("messages") else "",
        }

    def fix_results_mismatch(self, code_path: str, expected: dict, actual: dict) -> dict:
        """
        Fix code when results don't match expected values.

        Args:
            code_path: Path to code
            expected: Expected results from paper
            actual: Actual results from implementation

        Returns:
            Fix results
        """
        task = f"""Fix results mismatch in {code_path}:

Expected results:
{expected}

Actual results:
{actual}

Investigation:
1. Calculate the differences
2. Identify potential causes:
   - Algorithm implementation errors
   - Hyperparameter differences
   - Data preprocessing differences
   - Random seed issues
   - Numerical precision issues
3. Review the paper's algorithm description
4. Check implementation against paper
5. Fix identified issues
6. Re-run and verify

Provide analysis and fixes."""

        messages = [HumanMessage(content=task)]
        result = self.agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 50}
        )

        return {
            "mismatch_analysis": normalize_message_content(result.get("messages", [])[-1].content) if result.get("messages") else "",
        }

    def iterative_debug(self, code_path: str, max_iterations: int = 5) -> dict:
        """
        Iteratively debug code until it works or max iterations reached.

        Args:
            code_path: Path to code
            max_iterations: Maximum debug iterations

        Returns:
            Final debug results
        """
        task = f"""Iteratively debug {code_path} until it works:

Process (max {max_iterations} iterations):
1. Run the code
2. If it works, report success
3. If errors occur:
   a. Analyze the error
   b. Apply a fix
   c. Test the fix
   d. Repeat

After each iteration, report:
- What error was found
- What fix was applied
- Current status

Continue until code works or max iterations reached."""

        messages = [HumanMessage(content=task)]
        result = self.agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 50}
        )

        return {
            "iterations": max_iterations,
            "debug_log": normalize_message_content(result.get("messages", [])[-1].content) if result.get("messages") else "",
            "success": self._check_success(result),
        }

    def _parse_debug_result(self, result: dict) -> dict:
        """Parse debugging results."""
        messages = result.get("messages", [])

        debug_result = {
            "root_cause": "",
            "fixes_applied": [],
            "code_fixed": False,
            "report": "",
        }

        # Get final output
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content:
                content_str = normalize_message_content(msg.content)
                debug_result["report"] = content_str
                if "fixed" in content_str.lower() or "success" in content_str.lower():
                    debug_result["code_fixed"] = True
                break

        return debug_result

    def _check_success(self, result: dict) -> bool:
        """Check if debugging was successful."""
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, 'content'):
                content = normalize_message_content(msg.content).lower()
                if "success" in content or "works" in content or "fixed" in content:
                    return True
        return False
