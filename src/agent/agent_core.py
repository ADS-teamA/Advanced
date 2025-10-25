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
    from ..utils.security import sanitize_error
except ImportError:
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from cost_tracker import get_global_tracker
    from utils.security import sanitize_error

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
            logger.debug(f"Prompt caching: {config.enable_prompt_caching}")

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
            f"tools={len(self.registry)}, streaming={config.cli_config.enable_streaming}, "
            f"caching={config.enable_prompt_caching}"
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
        text = re.sub(r"#+\s+", "", text)

        # Remove markdown bold/italic (**text**, *text*)
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)

        # Replace multiple newlines with space
        text = re.sub(r"\n+", " ", text)

        # Replace multiple spaces with single space
        text = re.sub(r"\s+", " ", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def initialize_with_documents(self):
        """
        Initialize conversation by providing available documents.

        Adds document list to conversation history so agent knows what documents
        are available for search. Tool definitions are provided separately via
        the API 'tools' parameter and cached for efficiency.

        Note: Tool list is NOT included here to avoid duplication - Claude receives
        tool definitions directly via API parameter with prompt caching.
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

            # Build document list message (ONLY documents, NOT tools)
            # Tools are provided via API 'tools' parameter and cached separately
            doc_list_text = f"Available documents in the system ({count}):\n\n"
            for doc in documents:
                doc_id = doc.get("id", "Unknown")
                summary = doc.get("summary", "No summary")

                # Clean summary text
                summary = self._clean_summary_text(summary)

                # Truncate to first sentence or 150 chars
                if len(summary) > 150:
                    # Try to cut at sentence boundary
                    sentence_end = summary.find(". ", 0, 150)
                    if sentence_end > 50:  # Found reasonable sentence
                        summary = summary[: sentence_end + 1]
                    else:
                        summary = summary[:150] + "..."

                doc_list_text += f"- {doc_id}: {summary}\n"

            # Add footer explaining initialization
            init_message = doc_list_text
            init_message += "\n(These are the documents available in the system. Use your tools to search and analyze them.)"

            # Add as first message in conversation history
            self.conversation_history.append({"role": "user", "content": init_message})

            # Add simple acknowledgment from assistant
            self.conversation_history.append(
                {
                    "role": "assistant",
                    "content": "I understand. I have access to these documents and will use the appropriate tools to search and analyze them.",
                }
            )

            self._initialized_with_documents = True
            logger.info(f"Initialized conversation with {count} documents")

        except (FileNotFoundError, PermissionError) as e:
            logger.warning(f"Document initialization failed - file access issue: {e}")
            # Expected - vector store may be empty or permissions issue
        except KeyError as e:
            logger.error(f"Invalid metadata structure during initialization: {e}")
            # This indicates a data corruption issue - log as error
        except Exception as e:
            logger.error(
                f"Unexpected error initializing documents: {sanitize_error(e)}", exc_info=True
            )
            # Log as error for unexpected issues, but don't crash - continue without initialization

    def reset_conversation(self):
        """Reset conversation history and reinitialize with documents."""
        self.conversation_history = []
        self.tool_call_history = []
        self._initialized_with_documents = False
        logger.info("Conversation reset")

        # Reinitialize with documents and tools list
        self.initialize_with_documents()
        logger.info("Agent reinitialized with documents after reset")

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
        - User IS notified when history is trimmed (transparency)
        - Claude loses access to earlier conversation context
        - May break multi-turn reasoning across >50 message exchanges
        - Tool results in trimmed messages are lost permanently
        - Trimming happens BEFORE Claude API call (line 153)

        NOTES:
        - Each "message" may contain multiple tool results (lines 270, 344)
        - Actual context size varies significantly per message
        - Keeps the last MAX_HISTORY_MESSAGES messages
        - Adds notification message to conversation explaining trimming
        """
        if len(self.conversation_history) > MAX_HISTORY_MESSAGES:
            old_len = len(self.conversation_history)
            messages_removed = old_len - MAX_HISTORY_MESSAGES
            self.conversation_history = self.conversation_history[-MAX_HISTORY_MESSAGES:]

            logger.warning(
                f"Conversation history trimmed: {old_len} → {MAX_HISTORY_MESSAGES} messages "
                f"({messages_removed} messages removed)"
            )

            # CRITICAL: Notify user about trimming
            # Add a system message explaining what happened
            system_notice = {
                "role": "assistant",
                "content": f"[Note: Conversation history was trimmed. I can only access the last {MAX_HISTORY_MESSAGES} messages. Earlier context ({messages_removed} messages) is no longer available.]",
            }
            # Insert notice at beginning of trimmed history so Claude sees it
            self.conversation_history.insert(0, system_notice)

    def _prepare_system_prompt_with_cache(self) -> List[Dict[str, Any]]:
        """
        Prepare system prompt with cache control for Anthropic prompt caching.

        Returns system prompt as structured format:
        [
            {
                "type": "text",
                "text": "...",
                "cache_control": {"type": "ephemeral"}  # If caching enabled
            }
        ]

        Returns:
            List of system prompt blocks
        """
        system_block = {"type": "text", "text": self.config.system_prompt}

        # Add cache control if enabled
        if self.config.enable_prompt_caching:
            system_block["cache_control"] = {"type": "ephemeral"}

        return [system_block]

    def _prepare_tools_with_cache(self, tools: List[Dict]) -> List[Dict]:
        """
        Add cache control to tools array (on last tool) for Anthropic prompt caching.

        Caching strategy:
        - Cache control on last tool in array
        - Reduces API costs by 90% for tool definitions
        - TTL: 5 minutes (Anthropic default)

        Args:
            tools: List of tool definitions

        Returns:
            Tools with cache control added (if enabled)
        """
        if not self.config.enable_prompt_caching or not tools:
            return tools

        # Deep copy to avoid modifying original
        import copy

        cached_tools = copy.deepcopy(tools)

        # Add cache control to last tool
        cached_tools[-1]["cache_control"] = {"type": "ephemeral"}

        return cached_tools

    def _add_cache_control_to_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Add cache control to specific messages for Anthropic prompt caching.

        Caching strategy:
        - Cache document initialization message (first 2 messages)
        - Allows cached context across conversations

        Args:
            messages: Conversation history

        Returns:
            Messages with cache control added (if enabled)
        """
        if not self.config.enable_prompt_caching or len(messages) < 2:
            return messages

        # Deep copy to avoid modifying original
        import copy

        cached_messages = copy.deepcopy(messages)

        # Add cache control to assistant's acknowledgment (2nd message after init)
        # This caches the document list and tool list initialization
        if len(cached_messages) >= 2 and cached_messages[1]["role"] == "assistant":
            # Convert content to structured format if it's a string
            if isinstance(cached_messages[1]["content"], str):
                cached_messages[1]["content"] = [
                    {
                        "type": "text",
                        "text": cached_messages[1]["content"],
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            elif isinstance(cached_messages[1]["content"], list):
                # If already structured, add cache_control to last block
                if cached_messages[1]["content"]:
                    if isinstance(cached_messages[1]["content"][-1], dict):
                        cached_messages[1]["content"][-1]["cache_control"] = {"type": "ephemeral"}

        return cached_messages

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

            # Prepare cached system prompt and tools (static)
            system_prompt = self._prepare_system_prompt_with_cache()
            cached_tools = self._prepare_tools_with_cache(tools)

            while True:
                # Update cached messages with latest conversation history (includes tool results)
                cached_messages = self._add_cache_control_to_messages(self.conversation_history)

                with self.client.messages.stream(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    system=system_prompt,
                    messages=cached_messages,
                    tools=cached_tools,
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

                    # Track cost (including cache statistics if available)
                    cache_creation = getattr(final_message.usage, "cache_creation_input_tokens", 0)
                    cache_read = getattr(final_message.usage, "cache_read_input_tokens", 0)

                    self.tracker.track_llm(
                        provider="anthropic",
                        model=self.config.model,
                        input_tokens=final_message.usage.input_tokens,
                        output_tokens=final_message.usage.output_tokens,
                        operation="agent",
                        cache_creation_tokens=cache_creation,
                        cache_read_tokens=cache_read,
                    )

                    # Log cache hit info if debug mode
                    if self.config.debug_mode and (cache_creation > 0 or cache_read > 0):
                        logger.debug(
                            f"Cache usage: {cache_read} tokens read, {cache_creation} tokens created"
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

                        # Estimate cost of tool result (tokens added to context)
                        # Tool results are sent back to Claude in the next API call as input tokens
                        estimated_cost = 0.0
                        if result.estimated_tokens > 0:
                            # Get input cost per 1M tokens for current model
                            try:
                                from ..cost_tracker import PRICING

                                model_pricing = PRICING.get("anthropic", {}).get(self.config.model)
                                if model_pricing:
                                    input_price_per_1m = model_pricing.get("input", 0.0)
                                    estimated_cost = (
                                        result.estimated_tokens / 1_000_000
                                    ) * input_price_per_1m
                            except (KeyError, ValueError, ImportError) as e:
                                logger.error(
                                    f"Failed to calculate tool result cost for {tool_name}: {e}"
                                )

                        # Log tool execution with cost estimate
                        logger.info(
                            f"Tool '{tool_name}' result: ~{result.estimated_tokens} tokens, "
                            f"~${estimated_cost:.6f} cost estimate"
                        )

                        # Check for tool failure and alert user
                        if not result.success:
                            logger.error(
                                f"Tool '{tool_name}' failed: {result.error}",
                                extra={"tool_input": tool_input, "metadata": result.metadata},
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
                                "estimated_tokens": result.estimated_tokens,
                                "estimated_cost": estimated_cost,
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
            logger.error(f"Claude API error: {sanitize_error(e)}")
            yield f"\n\n[❌ API Error: {sanitize_error(e)}]\n"
        except Exception as e:
            logger.error(f"Streaming failed: {sanitize_error(e)}", exc_info=True)
            yield f"\n\n[❌ Unexpected error: {type(e).__name__}: {sanitize_error(e)}]\n"

    def _process_non_streaming(self, tools: List[Dict]) -> str:
        """
        Process message without streaming (synchronous).

        Returns complete response after tool execution.
        """
        full_response_text = ""

        # Prepare cached system prompt and tools (static)
        system_prompt = self._prepare_system_prompt_with_cache()
        cached_tools = self._prepare_tools_with_cache(tools)

        try:
            while True:
                # Update cached messages with latest conversation history (includes tool results)
                cached_messages = self._add_cache_control_to_messages(self.conversation_history)

                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    system=system_prompt,
                    messages=cached_messages,
                    tools=cached_tools,
                )

                # Track cost (including cache statistics if available)
                cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0)
                cache_read = getattr(response.usage, "cache_read_input_tokens", 0)

                self.tracker.track_llm(
                    provider="anthropic",
                    model=self.config.model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    operation="agent",
                    cache_creation_tokens=cache_creation,
                    cache_read_tokens=cache_read,
                )

                # Log cache hit info if debug mode
                if self.config.debug_mode and (cache_creation > 0 or cache_read > 0):
                    logger.debug(
                        f"Cache usage: {cache_read} tokens read, {cache_creation} tokens created"
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

                            # Estimate cost of tool result (tokens added to context)
                            # Tool results are sent back to Claude in the next API call as input tokens
                            estimated_cost = 0.0
                            if result.estimated_tokens > 0:
                                # Get input cost per 1M tokens for current model
                                try:
                                    from ..cost_tracker import PRICING

                                    model_pricing = PRICING.get("anthropic", {}).get(
                                        self.config.model
                                    )
                                    if model_pricing:
                                        input_price_per_1m = model_pricing.get("input", 0.0)
                                        estimated_cost = (
                                            result.estimated_tokens / 1_000_000
                                        ) * input_price_per_1m
                                except (KeyError, ValueError, ImportError) as e:
                                    logger.error(
                                        f"Failed to calculate tool result cost for {tool_name}: {e}"
                                    )

                            # Log tool execution with cost estimate
                            logger.info(
                                f"Tool '{tool_name}' result: ~{result.estimated_tokens} tokens, "
                                f"~${estimated_cost:.6f} cost estimate"
                            )

                            # Check for tool failure and log error
                            if not result.success:
                                error_msg = f"⚠️  Tool '{tool_name}' failed: {result.error}"
                                logger.error(
                                    f"Tool '{tool_name}' failed: {result.error}",
                                    extra={"tool_input": tool_input, "metadata": result.metadata},
                                )
                                # Show error to user in non-streaming mode
                                full_response_text += f"\n[{error_msg}]\n"

                            # Track in history
                            self.tool_call_history.append(
                                {
                                    "tool_name": tool_name,
                                    "input": tool_input,
                                    "success": result.success,
                                    "execution_time_ms": result.execution_time_ms,
                                    "estimated_tokens": result.estimated_tokens,
                                    "estimated_cost": estimated_cost,
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

        except anthropic.APITimeoutError as e:
            logger.error(f"API timeout in non-streaming mode: {e}")
            return "[⚠️  API timeout - response incomplete. Please try again.]"
        except anthropic.RateLimitError as e:
            logger.error(f"Rate limit hit: {e}")
            return "[⚠️  Rate limit exceeded - please wait a moment and try again.]"
        except anthropic.APIError as e:
            logger.error(f"Claude API error: {sanitize_error(e)}")
            return f"[❌ API Error: {sanitize_error(e)}]"
        except Exception as e:
            logger.error(f"Non-streaming processing failed: {sanitize_error(e)}", exc_info=True)
            return f"[❌ Unexpected error: {type(e).__name__}: {sanitize_error(e)}]"

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
