"""
Agent Core - Claude SDK Orchestration

Main agent loop with:
- Claude SDK integration
- Tool execution
- Conversation management
- Streaming responses
"""

import json
import logging
from typing import Any, Dict, Generator, List, Optional

import anthropic

from .config import AgentConfig
from .tools.base import ToolResult
from .tools.registry import get_registry

logger = logging.getLogger(__name__)

# Configuration constants
MAX_HISTORY_MESSAGES = 50  # Keep last 50 messages to prevent unbounded memory growth
MAX_QUERY_LENGTH = 10000  # Maximum characters in a single query


class AgentCore:
    """
    Core agent orchestration using Claude SDK.

    Manages:
    - Claude SDK client
    - Conversation history
    - Tool execution loop
    - Streaming responses
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize agent core.

        Args:
            config: AgentConfig instance
        """
        self.config = config

        if config.debug_mode:
            logger.debug("Initializing AgentCore...")
            logger.debug(f"Model: {config.model}")
            logger.debug(f"Max tokens: {config.max_tokens}")
            logger.debug(f"Temperature: {config.temperature}")

        # Validate config
        config.validate()

        # Initialize Claude SDK client
        if config.debug_mode:
            logger.debug("Initializing Claude SDK client...")
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

        # Get tool registry
        if config.debug_mode:
            logger.debug("Getting tool registry...")
        self.registry = get_registry()

        # Conversation history
        self.conversation_history: List[Dict[str, Any]] = []
        self.tool_call_history: List[Dict[str, Any]] = []

        logger.info(
            f"AgentCore initialized: model={config.model}, "
            f"tools={len(self.registry)}, streaming={config.cli_config.enable_streaming}"
        )

        if config.debug_mode:
            tool_list = list(self.registry._tool_classes.keys())
            logger.debug(f"Available tools: {tool_list}")

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
        self.tool_call_history = []
        logger.info("Conversation reset")

    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        return {
            "message_count": len(self.conversation_history),
            "tool_calls": len(self.tool_call_history),
            "tools_used": list(set(t["tool_name"] for t in self.tool_call_history)),
        }

    def _trim_history(self):
        """
        Trim conversation history to prevent unbounded memory growth.

        IMPORTANT IMPLICATIONS:
        - User is NOT notified when history is trimmed (silent truncation)
        - Claude loses access to earlier conversation context
        - May break multi-turn reasoning across >50 message exchanges
        - Tool results in trimmed messages are lost permanently
        - Trimming happens BEFORE processing current message (line 143)

        NOTES:
        - Each "message" may contain multiple tool results (lines 270, 344)
        - Actual context size varies significantly per message
        - Keeps the last MAX_HISTORY_MESSAGES messages
        """
        if len(self.conversation_history) > MAX_HISTORY_MESSAGES:
            old_len = len(self.conversation_history)
            self.conversation_history = self.conversation_history[-MAX_HISTORY_MESSAGES:]
            logger.info(
                f"Trimmed conversation history: {old_len} → {len(self.conversation_history)} messages"
            )

    def process_message(
        self, user_message: str, stream: bool = None
    ) -> Generator[str, None, None] | str:
        """
        Process user message with full agent loop.

        Flow:
        1. Validate query length
        2. Add user message to history
        3. Trim history if needed
        4. Call Claude with tools
        5. Execute tools if requested
        6. Get final answer
        7. Stream or return response

        Args:
            user_message: User's question/request
            stream: Enable streaming (default from config)

        Returns:
            Generator of text chunks (if streaming) or full text

        Raises:
            ValueError: If query is too long
        """
        # Validate query length (prevent DoS via huge queries)
        if len(user_message) > MAX_QUERY_LENGTH:
            raise ValueError(
                f"Query too long ({len(user_message)} chars). "
                f"Maximum length is {MAX_QUERY_LENGTH} characters."
            )

        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})

        # Trim history to prevent unbounded growth
        self._trim_history()

        # Determine streaming
        if stream is None:
            stream = self.config.cli_config.enable_streaming

        # Get Claude SDK tools
        tools = self.registry.get_claude_sdk_tools()

        logger.info(
            f"Processing message (streaming={stream}, tools={len(tools)}): "
            f"{user_message[:100]}..."
        )

        if stream:
            return self._process_streaming(tools)
        else:
            return self._process_non_streaming(tools)

    def _process_streaming(self, tools: List[Dict]) -> Generator[str, None, None]:
        """
        Process message with streaming.

        Yields text chunks as they arrive from Claude.
        Handles API errors gracefully to avoid incomplete responses.
        """
        try:
            # Import anthropic for error handling
            import anthropic

            while True:
                with self.client.messages.stream(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=self.config.system_prompt,
                messages=self.conversation_history,
                tools=tools,
            ) as stream:
                # Collect assistant message
                assistant_message = {"role": "assistant", "content": []}
                tool_uses = []

                # Stream text and collect tool uses
                for event in stream:
                    if event.type == "content_block_start":
                        if event.content_block.type == "text":
                            assistant_message["content"].append({"type": "text", "text": ""})

                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            # Stream text to user
                            yield event.delta.text

                            # Add to message content
                            for block in assistant_message["content"]:
                                if block["type"] == "text":
                                    block["text"] += event.delta.text
                                    break

                    elif event.type == "content_block_stop":
                        if (
                            hasattr(event, "content_block")
                            and event.content_block.type == "tool_use"
                        ):
                            tool_uses.append(event.content_block)
                            assistant_message["content"].append(
                                {
                                    "type": "tool_use",
                                    "id": event.content_block.id,
                                    "name": event.content_block.name,
                                    "input": event.content_block.input,
                                }
                            )

                # Get final message
                final_message = stream.get_final_message()

                # Extract tool uses from final message
                for block in final_message.content:
                    if block.type == "tool_use":
                        tool_uses.append(block)

                # Add assistant message to history
                if assistant_message["content"]:
                    self.conversation_history.append(assistant_message)

                # Check stop reason
                if final_message.stop_reason == "end_turn":
                    # Done - no tool calls
                    break

                elif final_message.stop_reason == "tool_use":
                    # Execute tools
                    yield "\n\n"  # Newline before tool execution

                    tool_results = []
                    for tool_use in tool_uses:
                        tool_name = tool_use.name
                        tool_input = tool_use.input

                        # Show tool call
                        if self.config.cli_config.show_tool_calls:
                            yield f"[Using {tool_name}...]\n"

                        # Execute tool
                        logger.info(f"Executing tool: {tool_name} with input {tool_input}")

                        result = self.registry.execute_tool(tool_name, **tool_input)

                        # Check for tool failure and alert user
                        if not result.success:
                            logger.error(
                                f"Tool '{tool_name}' failed: {result.error}",
                                extra={"tool_input": tool_input, "metadata": result.metadata}
                            )
                            # Alert user in streaming mode if show_tool_calls is enabled
                            if self.config.cli_config.show_tool_calls:
                                yield f"[⚠️  Tool '{tool_name}' failed: {result.error}]\n"

                        # Track in history
                        self.tool_call_history.append(
                            {
                                "tool_name": tool_name,
                                "input": tool_input,
                                "success": result.success,
                                "execution_time_ms": result.execution_time_ms,
                            }
                        )

                        # Format tool result for Claude
                        tool_result_content = self._format_tool_result(result)

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": tool_result_content,
                            }
                        )

                    # Add tool results to conversation
                    self.conversation_history.append({"role": "user", "content": tool_results})

                    # Continue loop to get final answer
                    yield "\n"  # Newline after tool execution

                else:
                    logger.warning(f"Unexpected stop reason: {final_message.stop_reason}")
                    break

        except anthropic.APITimeoutError as e:
            logger.error(f"API timeout during streaming: {e}")
            yield "\n\n[⚠️  API timeout - response incomplete. Please try again.]\n"
        except anthropic.RateLimitError as e:
            logger.error(f"Rate limit hit: {e}")
            yield "\n\n[⚠️  Rate limit exceeded - please wait a moment and try again.]\n"
        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            yield f"\n\n[❌ API Error: {e}]\n"
        except Exception as e:
            logger.error(f"Streaming failed: {e}", exc_info=True)
            yield f"\n\n[❌ Unexpected error: {type(e).__name__}: {e}]\n"

    def _process_non_streaming(self, tools: List[Dict]) -> str:
        """
        Process message without streaming (synchronous).

        Returns complete response after tool execution.
        """
        full_response_text = ""

        while True:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=self.config.system_prompt,
                messages=self.conversation_history,
                tools=tools,
            )

            # Add assistant message to history
            self.conversation_history.append({"role": "assistant", "content": response.content})

            # Extract text
            for block in response.content:
                if block.type == "text":
                    full_response_text += block.text

            # Check stop reason
            if response.stop_reason == "end_turn":
                break

            elif response.stop_reason == "tool_use":
                # Execute tools
                tool_results = []

                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input

                        logger.info(f"Executing tool: {tool_name}")

                        result = self.registry.execute_tool(tool_name, **tool_input)

                        # Check for tool failure and log error
                        if not result.success:
                            logger.error(
                                f"Tool '{tool_name}' failed: {result.error}",
                                extra={"tool_input": tool_input, "metadata": result.metadata}
                            )

                        # Track in history
                        self.tool_call_history.append(
                            {
                                "tool_name": tool_name,
                                "input": tool_input,
                                "success": result.success,
                                "execution_time_ms": result.execution_time_ms,
                            }
                        )

                        # Format tool result
                        tool_result_content = self._format_tool_result(result)

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": tool_result_content,
                            }
                        )

                # Add tool results to conversation
                self.conversation_history.append({"role": "user", "content": tool_results})

            else:
                logger.warning(f"Unexpected stop reason: {response.stop_reason}")
                break

        return full_response_text

    def _format_tool_result(self, result: ToolResult) -> str:
        """
        Format ToolResult for Claude SDK.

        Args:
            result: ToolResult from tool execution

        Returns:
            Formatted string for Claude
        """
        if not result.success:
            return json.dumps({"error": result.error, "metadata": result.metadata}, indent=2)

        # Format successful result
        formatted = {"data": result.data, "metadata": result.metadata}

        if result.citations:
            formatted["citations"] = result.citations

        return json.dumps(formatted, indent=2)
