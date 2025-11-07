"""Environment Setup Agent - Installs dependencies and prepares execution environment."""

from typing import Dict
from langchain_core.messages import HumanMessage
from ..tools.code_execution_tools import (
    read_file,
    list_directory,
    execute_shell_command,
    install_dependencies,
    check_python_compatibility,
    smart_install_dependencies,
)
from ..utils.llm_factory import create_llm
from ..utils.xml_tool_parser import create_xml_aware_agent_executor
from ..utils.logging_callback import LoggingCallbackHandler
from ..utils.message_utils import normalize_message_content
import os


class EnvironmentSetupAgent:
    """Agent for setting up execution environment."""

    def __init__(self, llm=None, callback=None, max_iterations=10):
        self.llm = llm or create_llm(temperature=0.1)
        self.callback = callback
        self.max_iterations = max_iterations

        self.system_prompt = """You set up Python execution environments with smart error handling.

CRITICAL RULES:
1. FIRST, check Python version compatibility using check_python_compatibility tool
2. If incompatible, STOP immediately and report the issue - DO NOT attempt installation
3. Use smart_install_dependencies for intelligent version fallback
4. Maximum {max_iterations} tool calls - be efficient
5. If same error occurs 3 times, STOP and report failure
6. IMPORTANT: After EVERY tool call, explain what you learned and what you'll do next

Priority order:
1. check_python_compatibility(repo_path) - CHECK THIS FIRST!
   - After receiving results, explain the compatibility status
2. If compatible, use smart_install_dependencies(repo_path)
   - After receiving results, explain if installation succeeded or failed
3. If installation succeeds, verify with simple import test
4. Report success or failure with clear explanation

DO NOT waste iterations trying impossible installations.
ALWAYS provide reasoning text after receiving tool results.""".format(max_iterations=max_iterations)

        tools = [
            check_python_compatibility,
            smart_install_dependencies,
            read_file,
            list_directory,
            execute_shell_command,
            install_dependencies,
        ]

        # Detect if we're using vLLM (needs XML parsing) or native tool calling (Groq, etc.)
        use_xml_parser = self._should_use_xml_parser()

        if use_xml_parser:
            print("🔧 Using XML tool call parser for vLLM")
            # Update system prompt to instruct XML format
            self.system_prompt += """

When using tools, output them in this format:
<tool_call>
{"name": "tool_name", "arguments": {"arg1": "value1"}}
</tool_call>"""
            # Use XML-aware executor with callback
            self.agent_executor = create_xml_aware_agent_executor(
                self.llm,
                tools=tools,
                system_prompt=self.system_prompt,
                max_iterations=self.max_iterations,
                callbacks=[callback] if callback else None
            )
            self.use_xml = True
        else:
            print("🔧 Using native tool calling (ReAct agent)")
            # Use standard ReAct agent with native tool calling
            from langgraph.prebuilt import create_react_agent

            # Use the LLM as-is - don't unwrap! The wrapper handles multi-turn properly
            # Unwrapping was causing empty responses with some LLMs
            llm_for_react = self.llm

            # IMPORTANT: Don't pass custom prompt - it interferes with tool calling
            # Let ReAct use its default prompt which is optimized for tool use
            self.agent = create_react_agent(llm_for_react, tools=tools)
            self.use_xml = False

    def _should_use_xml_parser(self) -> bool:
        """
        Detect if we should use XML parser based on LLM configuration.

        Returns True only if explicitly enabled via USE_XML_TOOL_PARSER env var.
        GPT-OSS and most modern models support native tool calling, so XML is rarely needed.
        """
        # Explicit override via environment variable
        use_xml = os.getenv("USE_XML_TOOL_PARSER", "false").lower() == "true"

        if use_xml:
            print("🔧 XML tool parser explicitly enabled via USE_XML_TOOL_PARSER=true")
            return True

        # Default to native tool calling for modern models
        return False

    def setup_environment(self, code_path: str) -> Dict:
        """
        Install dependencies for a repository.

        Args:
            code_path: Path to repository

        Returns:
            Setup results with status and errors
        """
        # Prepend system context to task (since we can't override ReAct's prompt)
        task = f"""{self.system_prompt}

Task: Set up Python environment for repository at: {code_path}

STEP-BY-STEP INSTRUCTIONS (DO NOT SKIP STEPS):

Step 1: Check Python Compatibility (REQUIRED FIRST STEP)
   Tool call: check_python_compatibility(repo_path="{code_path}")

   If the result shows compatible=False:
   - STOP immediately
   - Report the incompatibility issue
   - Provide the suggestions from the tool
   - DO NOT attempt installation

Step 2: Install Dependencies (ONLY if Step 1 shows compatible=True)
   Tool call: smart_install_dependencies(repo_path="{code_path}")

   This tool will automatically:
   - Try original requirements
   - Fall back to relaxed versions if needed
   - Fall back to unpinned versions if still failing

   If this fails after all strategies, STOP and report failure.

Step 3: Verify Installation (ONLY if Step 2 succeeds)
   Tool call: execute_shell_command(command="python -c 'import sys; print(sys.version)'", cwd="{code_path}")

CRITICAL: Do NOT manually install dependencies. Use smart_install_dependencies tool which handles fallbacks automatically.

Maximum {self.max_iterations} iterations total. Be efficient and stop early if incompatible."""

        if self.use_xml:
            # XML parser mode for vLLM
            result = self.agent_executor(task)
        else:
            # Native tool calling mode (ReAct agent) for Groq, OpenAI, etc.
            messages = [HumanMessage(content=task)]
            callback = LoggingCallbackHandler(verbose=True)

            try:
                result = self.agent.invoke(
                    {"messages": messages},
                    config={
                        "recursion_limit": self.max_iterations * 2,  # Allow some overhead
                        "callbacks": [callback]
                    }
                )

                # Debug: Print what we got back
                print(f"\n🔍 DEBUG: Agent returned result with {len(result.get('messages', []))} messages")
                for i, msg in enumerate(result.get('messages', [])):
                    print(f"  Message {i}: type={type(msg).__name__}, has_content={hasattr(msg, 'content')}")
                    if hasattr(msg, 'tool_calls'):
                        print(f"    tool_calls: {len(msg.tool_calls) if msg.tool_calls else 0}")

            except Exception as e:
                print(f"\n❌ Agent execution failed: {e}")
                import traceback
                traceback.print_exc()
                result = {"messages": [], "error": str(e)}

        return self._parse_setup_result(result, code_path)

    def _parse_setup_result(self, result: Dict, code_path: str) -> Dict:
        """Extract setup status from agent result."""
        messages = result.get("messages", [])

        setup_info = {
            "success": False,
            "dependencies_found": False,
            "dependencies_installed": False,
            "python_compatible": True,
            "compatibility_warnings": [],
            "errors": [],
            "report": "",
            "strategy_used": None
        }

        print(f"\n🔍 DEBUG: Parsing {len(messages)} messages from agent")

        # CRITICAL: First check ToolMessage for actual tool return values
        # This is more reliable than parsing AI message text
        for msg in messages:
            # Check if this is a ToolMessage from smart_install_dependencies
            if hasattr(msg, 'name') and msg.name == 'smart_install_dependencies':
                if hasattr(msg, 'content') and msg.content:
                    try:
                        import json
                        tool_result = json.loads(msg.content)
                        if tool_result.get("success"):
                            setup_info["success"] = True
                            setup_info["dependencies_installed"] = True
                            setup_info["strategy_used"] = tool_result.get("strategy_used", "unknown")
                            print(f"✅ Detected successful installation from tool result")
                            # Still parse other messages for report, but we know it succeeded
                            break
                    except (json.JSONDecodeError, AttributeError):
                        pass  # Fall back to text parsing

        # Check for Python compatibility issues and other info in messages
        for msg in messages:
            if hasattr(msg, 'content') and msg.content:
                # Normalize content to string (handles both simple strings and list format)
                content_str = normalize_message_content(msg.content)
                content = content_str.lower()

                # Check for compatibility warnings
                if "incompatible" in content or "not compatible" in content:
                    setup_info["python_compatible"] = False
                    setup_info["compatibility_warnings"].append(content_str[:300])

                # Check for dependency files found
                if "requirements.txt" in content or "setup.py" in content or "pyproject.toml" in content:
                    setup_info["dependencies_found"] = True

                # Fallback: Check for successful installation in text (if tool message not found)
                if not setup_info["success"]:
                    if ("successfully installed" in content or
                        "requirement already satisfied" in content or
                        "strategy_used" in content or
                        "set up successfully" in content):  # Added this!
                        setup_info["dependencies_installed"] = True
                        setup_info["success"] = True

                        # Extract strategy used if mentioned
                        if "original" in content:
                            setup_info["strategy_used"] = "original"
                        elif "relaxed" in content:
                            setup_info["strategy_used"] = "relaxed_versions"
                        elif "unpinned" in content:
                            setup_info["strategy_used"] = "unpinned_versions"

                # Check for errors
                if "error" in content or "failed" in content or "stopped early" in content:
                    # Don't count "no error" or success messages as errors
                    if "no error" not in content and "successfully" not in content:
                        setup_info["errors"].append(content_str[:300])

        # If Python is incompatible, mark as failed
        # BUT: Don't override if installation already succeeded (compatibility may have been resolved)
        if not setup_info["python_compatible"]:
            # Only override if we didn't already confirm success from tool result
            if not setup_info.get("dependencies_installed"):
                setup_info["success"] = False
                setup_info["dependencies_installed"] = False
                print(f"⚠️  Python incompatible and no successful installation detected")
            else:
                # Installation succeeded despite initial compatibility concerns - trust the tool result
                print(f"ℹ️  Initial compatibility warnings found, but installation succeeded")
                setup_info["python_compatible"] = True  # Installation resolved the issue

        # Get final report
        if messages:
            final_msg = messages[-1]
            if hasattr(final_msg, 'content'):
                setup_info["report"] = normalize_message_content(final_msg.content)
            else:
                setup_info["report"] = str(final_msg)

        # Add compatibility info to report if there were warnings
        if setup_info["compatibility_warnings"]:
            setup_info["report"] = "\n".join([
                "Python Compatibility Issues:",
                *setup_info["compatibility_warnings"],
                "",
                "Original Report:",
                setup_info["report"]
            ])

        # Final debug output
        print(f"🔍 DEBUG: Final parse result:")
        print(f"  success={setup_info['success']}")
        print(f"  dependencies_installed={setup_info['dependencies_installed']}")
        print(f"  python_compatible={setup_info['python_compatible']}")
        print(f"  strategy_used={setup_info.get('strategy_used')}")

        return setup_info
