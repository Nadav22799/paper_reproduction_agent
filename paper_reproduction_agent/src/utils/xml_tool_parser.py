"""XML Tool Call Parser - Extracts and executes tool calls from XML format."""

import json
import re
from typing import Dict, List, Any, Optional


def extract_tool_calls_from_xml(text: str) -> List[Dict[str, Any]]:
    """
    Extract tool calls from XML-formatted text.

    Expected format:
    <tool_call>
    {"name": "tool_name", "arguments": {...}}
    </tool_call>

    Args:
        text: Text potentially containing <tool_call> tags

    Returns:
        List of tool call dictionaries with 'name' and 'arguments'
    """
    tool_calls = []

    # Find all <tool_call>...</tool_call> blocks
    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        try:
            # Parse the JSON inside the tool_call tag
            tool_data = json.loads(match)

            if 'name' in tool_data:
                tool_calls.append({
                    'name': tool_data['name'],
                    'arguments': tool_data.get('arguments', {})
                })
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse tool call JSON: {match[:100]}")
            print(f"   Error: {e}")
            continue

    return tool_calls


def execute_tool_calls(tool_calls: List[Dict[str, Any]], tools: List) -> List[Dict[str, Any]]:
    """
    Execute a list of tool calls.

    Args:
        tool_calls: List of tool call dictionaries
        tools: List of available tool objects

    Returns:
        List of execution results
    """
    results = []

    # Create a mapping of tool names to tool objects
    tool_map = {tool.name: tool for tool in tools}

    for tool_call in tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['arguments']

        if tool_name not in tool_map:
            results.append({
                'tool': tool_name,
                'success': False,
                'error': f"Tool '{tool_name}' not found"
            })
            continue

        try:
            # Execute the tool
            tool = tool_map[tool_name]
            result = tool.invoke(tool_args)

            results.append({
                'tool': tool_name,
                'success': True,
                'result': result
            })

        except Exception as e:
            results.append({
                'tool': tool_name,
                'success': False,
                'error': str(e)
            })

    return results


def create_xml_aware_agent_executor(llm, tools: List, system_prompt: str, max_iterations: int = 10, callbacks=None):
    """
    Create an agent executor that can handle XML-formatted tool calls.

    This is a custom implementation that doesn't rely on LangChain's ReAct agent,
    which expects native tool calling support from the model.

    Args:
        llm: Language model
        tools: List of available tools
        system_prompt: System prompt for the agent
        max_iterations: Maximum number of iterations
        callbacks: Optional callbacks for logging

    Returns:
        A callable agent executor function
    """
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    def agent_executor(task: str) -> Dict[str, Any]:
        """
        Execute the agent task.

        Args:
            task: Task description

        Returns:
            Execution result with messages and final answer
        """
        messages = [SystemMessage(content=system_prompt)]
        messages.append(HumanMessage(content=task))

        all_messages = []
        iterations = 0

        while iterations < max_iterations:
            iterations += 1

            # Get LLM response with callbacks
            try:
                if callbacks:
                    response = llm.invoke(messages, config={"callbacks": callbacks})
                else:
                    response = llm.invoke(messages)
            except Exception as e:
                print(f"\n❌ LLM invocation failed: {e}")
                return {
                    'messages': all_messages,
                    'final_answer': f"Error: {str(e)}",
                    'iterations': iterations,
                    'error': str(e)
                }
            response_text = response.content if hasattr(response, 'content') else str(response)

            all_messages.append(response)

            # Check if there are tool calls in the response
            tool_calls = extract_tool_calls_from_xml(response_text)

            if not tool_calls:
                # No more tool calls, we're done
                # Extract final answer (text without <think> or <tool_call> tags)
                final_answer = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
                final_answer = re.sub(r'<tool_call>.*?</tool_call>', '', final_answer, flags=re.DOTALL)
                final_answer = final_answer.strip()

                return {
                    'messages': all_messages,
                    'final_answer': final_answer,
                    'iterations': iterations
                }

            # Execute the tool calls
            print(f"\n🔧 Iteration {iterations}: Executing {len(tool_calls)} tool call(s)")
            execution_results = execute_tool_calls(tool_calls, tools)

            # Format tool results as a message
            tool_results_text = "Tool execution results:\n\n"
            for i, (tool_call, result) in enumerate(zip(tool_calls, execution_results), 1):
                tool_results_text += f"{i}. {tool_call['name']}:\n"
                if result['success']:
                    tool_results_text += f"   Result: {result['result']}\n\n"
                else:
                    tool_results_text += f"   Error: {result['error']}\n\n"

            # Add tool results to conversation
            messages.append(AIMessage(content=response_text))
            messages.append(HumanMessage(content=tool_results_text))
            all_messages.append(HumanMessage(content=tool_results_text))

        # Max iterations reached
        print(f"⚠️ Max iterations ({max_iterations}) reached")
        return {
            'messages': all_messages,
            'final_answer': "Max iterations reached",
            'iterations': iterations
        }

    return agent_executor
