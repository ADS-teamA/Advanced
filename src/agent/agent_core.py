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

try:
    from ..cost_tracker import get_global_tracker
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from cost_tracker import get_global_tracker

logger = logging.getLogger(__name__)

# Configuration constants
MAX_HISTORY_MESSAGES = 50  # Keep last 50 messages to prevent unbounded memory growth
MAX_QUERY_LENGTH = 10000  # Maximum characters in a single query

# ANSI color codes for terminal output
COLOR_BLUE = "\033[1;34m"  # Bold blue for tool calls and debug
COLOR_RESET = "\033[0m"  # Reset color


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

        # Initialize cost tracker
        self.tracker = get_global_tracker()

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

        # Flag to track if initialized
        self._initialized_with_documents = False

    def _clean_summary_text(self, text: str) -> str:
        """
        Clean summary text for conversation history.

        Removes:
        - HTML entities (&lt;, &gt;, etc.)
        - Markdown formatting (##, **, etc.)
        - Extra whitespace and newlines
        """
        import html
        import re

        # Unescape HTML entities
        text = html.unescape(text)

        # Remove markdown headers (## Header)
        text = re.sub(r'#+\s+', '', text)

        # Remove markdown bold/italic (**text**, *text*)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)

        # Replace multiple newlines with space
        text = re.sub(r'\n+', ' ', text)

        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def initialize_with_documents(self):
        """
        Initialize conversation by calling get_document_list and list_available_tools.

        Adds document list and tool list to conversation history so agent has context
        about available documents and tools before first user query.
        """
        if self._initialized_with_documents:
            return  # Already initialized

        try:
            # Get document list tool
            doc_list_tool = self.registry.get_tool("get_document_list")
            if not doc_list_tool:
                logger.warning("get_document_list tool not available for initialization")
                return

            # Execute document list tool
            doc_result = doc_list_tool.execute()

            if not doc_result.success or not doc_result.data:
                logger.warning("Failed to get document list for initialization")
                return

            documents = doc_result.data.get("documents", [])
            count = doc_result.data.get("count", 0)

            if count == 0:
                logger.info("No documents available for initialization")
                return

            # Build document list message
            doc_list_text = f"Available documents in the system ({count}):\n\n"
            for doc in documents:
                doc_id = doc.get("id", "Unknown")
                summary = doc.get("summary", "No summary")

                # Clean summary text
                summary = self._clean_summary_text(summary)

                # Truncate to first sentence or 150 chars
                if len(summary) > 150:
                    # Try to cut at sentence boundary
                    sentence_end = summary.find('. ', 0, 150)
                    if sentence_end > 50:  # Found reasonable sentence
                        summary = summary[:sentence_end + 1]
                    else:
                        summary = summary[:150] + "..."

                doc_list_text += f"- {doc_id}: {summary}\n"

            # Get tool list tool
            tool_list_tool = self.registry.get_tool("list_available_tools")
            if not tool_list_tool:
                logger.warning("list_available_tools tool not available for initialization")
                # Continue without tools list
                tool_list_text = ""
            else:
                # Execute tool list tool
                tool_result = tool_list_tool.execute()

                if not tool_result.success or not tool_result.data:
                    logger.warning("Failed to get tool list for initialization")
                    tool_list_text = ""
                else:
                    tools = tool_result.data.get("tools", [])
                    tool_count = len(tools)

                    # Build tool list message (summary only, not full details)
                    tool_list_text = f"\n\nAvailable tools ({tool_count}):\n\n"

                    # Group by tier
                    tier1_tools = [t for t in tools if "Tier 1" in t.get("tier", "")]
                    tier2_tools = [t for t in tools if "Tier 2" in t.get("tier", "")]
                    tier3_tools = [t for t in tools if "Tier 3" in t.get("tier", "")]

                    if tier1_tools:
                        tool_list_text += "TIER 1 - Basic Retrieval (fast, <100ms):\n"
                        for tool in tier1_tools:
                            tool_list_text += f"  • {tool['name']}: {tool['description'][:80]}...\n"

                    if tier2_tools:
                        tool_list_text += "\nTIER 2 - Advanced Retrieval (500-1000ms):\n"
                        for tool in tier2_tools:
                            tool_list_text += f"  • {tool['name']}: {tool['description'][:80]}...\n"

                    if tier3_tools:
                        tool_list_text += "\nTIER 3 - Analysis & Insights (1-3s):\n"
                        for tool in tier3_tools:
                            tool_list_text += f"  • {tool['name']}: {tool['description'][:80]}...\n"

            # Combine messages
            init_message = doc_list_text + tool_list_text
            init_message += "\n\n(This is system initialization - the user will now ask questions about these documents)"

            # Add as first message in conversation history
            self.conversation_history.append({
                "role": "user",
                "content": init_message
            })

            # Add simple acknowledgment from assistant
            self.conversation_history.append({
                "role": "assistant",
                "content": "I understand. I have access to these documents and will use the appropriate tools to search them and answer user questions."
            })

            self._initialized_with_documents = True
            logger.debug(f"Initialized conversation with {count} documents and {len(self.registry)} tools")

        except Exception as e:
            logger.warning(f"Could not initialize with documents: {e}")
            # Don't fail - just continue without initialization

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
        self.tool_call_history = []
        self._initialized_with_documents = False
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
        - Trimming happens BEFORE Claude API call (line 153)

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

                    # Track cost
                    self.tracker.track_llm(
                        provider="anthropic",
                        model=self.config.model,
                        input_tokens=final_message.usage.input_tokens,
                        output_tokens=final_message.usage.output_tokens,
                        operation="agent"
                    )

                    # Note: tool_uses already collected during streaming (lines 224-237)
                    # No need to extract from final_message again - would cause duplicates!

                    # Add assistant message to history
                    if assistant_message["content"]:
                        self.conversation_history.append(assistant_message)

                # Check stop reason (outside of with block but inside while)
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

                        # Show tool call (in blue)
                        if self.config.cli_config.show_tool_calls:
                            yield f"{COLOR_BLUE}[Using {tool_name}...]{COLOR_RESET}\n"

                        # Execute tool
                        logger.info(f"Executing tool: {tool_name} with input {tool_input}")

                        result = self.registry.execute_tool(tool_name, **tool_input)

                        # Check for tool failure and alert user
                        if not result.success:
                            logger.error(
                                f"Tool '{tool_name}' failed: {result.error}",
                                extra={"tool_input": tool_input, "metadata": result.metadata}
                            )
                            # Alert user in streaming mode if show_tool_calls is enabled (in blue)
                            if self.config.cli_config.show_tool_calls:
                                yield f"{COLOR_BLUE}[⚠️  Tool '{tool_name}' failed: {result.error}]{COLOR_RESET}\n"

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

            # Track cost
            self.tracker.track_llm(
                provider="anthropic",
                model=self.config.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                operation="agent"
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
