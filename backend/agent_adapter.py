"""
Agent Adapter - Wraps existing AgentCore without modifying src/.

This adapter:
1. Imports AgentCore from src/agent/agent_core.py
2. Handles SSE event formatting
3. Tracks cost per message
4. Provides clean interface for FastAPI
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, Optional

from src.agent.agent_core import AgentCore
from src.agent.config import AgentConfig
from src.agent.tools.registry import get_registry
from src.context_assembly import CitationFormat, ContextAssembler
from src.cost_tracker import get_global_tracker, reset_global_tracker
from src.embedding_generator import EmbeddingGenerator
from src.hybrid_search import HybridVectorStore
from src.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class AgentAdapter:
    """
    Adapter wrapping AgentCore for web frontend.

    Responsibilities:
    - Initialize AgentCore with config
    - Convert streaming events to SSE format
    - Track costs per request
    - Provide model switching capability
    """

    def __init__(self, vector_store_path: Optional[Path] = None, model: Optional[str] = None):
        """
        Initialize agent adapter.

        Args:
            vector_store_path: Path to vector store (default: ../vector_db from backend/)
            model: Model to use (default: from config)
        """
        # Load config from environment
        config_overrides = {}

        # Set default vector_store_path relative to project root (parent of backend/)
        if vector_store_path:
            config_overrides["vector_store_path"] = vector_store_path
        else:
            # Backend runs from backend/, so vector_db is in parent directory
            project_root = Path(__file__).parent.parent
            config_overrides["vector_store_path"] = project_root / "vector_db"

        if model:
            config_overrides["model"] = model

        self.config = AgentConfig.from_env(**config_overrides)

        # Validate config
        try:
            self.config.validate()
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            raise

        # Initialize pipeline components (same as CLI)
        logger.info("Loading vector store...")
        vector_store = HybridVectorStore.load(str(self.config.vector_store_path))

        logger.info("Initializing embedder...")
        embedder = EmbeddingGenerator()

        # Track degraded components (instance variable for health endpoint)
        self.degraded_components = []

        # Initialize reranker (optional, lazy load)
        reranker = None
        if self.config.tool_config.enable_reranking:
            if not self.config.tool_config.lazy_load_reranker:
                logger.info("Loading reranker...")
                try:
                    reranker = CrossEncoderReranker(
                        model_name=self.config.tool_config.reranker_model
                    )
                except (ImportError, ModuleNotFoundError) as e:
                    logger.error(
                        f"Reranker dependencies missing: {e}. "
                        f"Install with: pip install sentence-transformers"
                    )
                    self.config.tool_config.enable_reranking = False
                    self.degraded_components.append({
                        "component": "reranker",
                        "error": f"Missing dependencies: {e}"
                    })
                except (FileNotFoundError, ValueError) as e:
                    logger.error(
                        f"Reranker configuration error: {e}. "
                        f"Check model name '{self.config.tool_config.reranker_model}' in config."
                    )
                    self.config.tool_config.enable_reranking = False
                    self.degraded_components.append({
                        "component": "reranker",
                        "error": f"Configuration error: {e}"
                    })
                except RuntimeError as e:
                    if "CUDA" in str(e) or "GPU" in str(e):
                        logger.warning(
                            f"GPU unavailable for reranker: {e}. This is expected on CPU-only systems."
                        )
                        self.config.tool_config.enable_reranking = False
                        self.degraded_components.append({
                            "component": "reranker",
                            "error": "GPU unavailable (CPU-only mode)"
                        })
                    else:
                        logger.critical(f"Unexpected runtime error loading reranker: {e}", exc_info=True)
                        self.config.tool_config.enable_reranking = False
                        self.degraded_components.append({
                            "component": "reranker",
                            "error": f"Runtime error: {e}"
                        })
                except Exception as e:
                    # Catch-all for truly unexpected errors
                    logger.critical(
                        f"Unexpected error loading reranker: {type(e).__name__}: {e}",
                        exc_info=True
                    )
                    self.config.tool_config.enable_reranking = False
                    self.degraded_components.append({
                        "component": "reranker",
                        "error": f"Unexpected error: {type(e).__name__}"
                    })
            else:
                logger.info("Reranker set to lazy load")

        # Load knowledge graph (optional) - UNIFIED WITH CLI
        knowledge_graph = None
        graph_retriever = None
        if self.config.enable_knowledge_graph and self.config.knowledge_graph_path:
            logger.info("Loading knowledge graph...")
            try:
                from src.graph.models import KnowledgeGraph
                from src.graph_retrieval import GraphEnhancedRetriever
                import os

                # Check if Neo4j backend is configured (same as CLI)
                kg_backend = os.getenv("KG_BACKEND", "simple").lower()

                if kg_backend == "neo4j":
                    # Use Neo4j via GraphAdapter (same as CLI)
                    logger.info("Using Neo4j backend...")
                    try:
                        from src.graph import Neo4jConfig
                        from src.agent.graph_adapter import GraphAdapter

                        neo4j_config = Neo4jConfig.from_env()
                        knowledge_graph = GraphAdapter.from_neo4j(neo4j_config)

                        # Get stats for display
                        entity_count = len(knowledge_graph.entities)
                        rel_count = len(knowledge_graph.relationships)

                        logger.info(
                            f"✓ Connected to Neo4j: {entity_count} entities, {rel_count} relationships"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to connect to Neo4j: {e}")
                        logger.info("Falling back to JSON...")
                        kg_backend = "simple"  # Fallback to JSON

                if kg_backend != "neo4j":
                    # Use JSON (original behavior)
                    kg_path = self.config.knowledge_graph_path

                    # Check if path is a directory (vector_db/) or a single file
                    if kg_path.is_dir():
                        # Prefer unified_kg.json if it exists
                        unified_kg_path = kg_path / "unified_kg.json"

                        if unified_kg_path.exists():
                            # Load unified KG (already deduplicated with cross-doc relationships)
                            logger.info(f"Loading unified knowledge graph from JSON...")
                            knowledge_graph = KnowledgeGraph.load_json(str(unified_kg_path))
                            logger.info(
                                f"Unified KG loaded: {len(knowledge_graph.entities)} entities, "
                                f"{len(knowledge_graph.relationships)} relationships"
                            )
                        else:
                            # Fallback: Load all *_kg.json files from directory (old behavior)
                            kg_files = sorted(kg_path.glob("*_kg.json"))
                            if not kg_files:
                                raise FileNotFoundError(
                                    f"No knowledge graph files (*_kg.json or unified_kg.json) found in {kg_path}"
                                )

                            logger.info(f"Found {len(kg_files)} knowledge graph files (unified_kg.json not found)")

                            # Load and merge all KG files (naive merge without deduplication)
                            knowledge_graph = None
                            total_entities = 0
                            total_relationships = 0

                            for kg_file in kg_files:
                                kg = KnowledgeGraph.load_json(str(kg_file))
                                if knowledge_graph is None:
                                    knowledge_graph = kg
                                else:
                                    # Merge graphs by combining entities and relationships
                                    knowledge_graph.entities.extend(kg.entities)
                                    knowledge_graph.relationships.extend(kg.relationships)

                                total_entities += len(kg.entities)
                                total_relationships += len(kg.relationships)
                                logger.debug(
                                    f"Loaded {kg_file.name}: {len(kg.entities)} entities, "
                                    f"{len(kg.relationships)} relationships"
                                )

                            logger.info(
                                f"Total: {total_entities} entities, "
                                f"{total_relationships} relationships (naive merge - consider building unified_kg.json)"
                            )
                    else:
                        # Load single file
                        knowledge_graph = KnowledgeGraph.load_json(str(self.config.knowledge_graph_path))

                graph_retriever = GraphEnhancedRetriever(
                    vector_store=vector_store, knowledge_graph=knowledge_graph
                )
            except (ImportError, ModuleNotFoundError) as e:
                logger.error(
                    f"Knowledge graph dependencies missing: {e}. "
                    f"Install with: pip install networkx"
                )
                self.config.enable_knowledge_graph = False
                self.degraded_components.append({
                    "component": "knowledge_graph",
                    "error": f"Missing dependencies: {e}"
                })
            except FileNotFoundError as e:
                logger.warning(
                    f"Knowledge graph file not found at: {self.config.knowledge_graph_path}. "
                    f"Run indexing with ENABLE_KNOWLEDGE_GRAPH=true to generate it: "
                    f"uv run python run_pipeline.py data/your_docs/"
                )
                self.config.enable_knowledge_graph = False
                self.degraded_components.append({
                    "component": "knowledge_graph",
                    "error": f"File not found - run indexing to generate"
                })
            except (ValueError, KeyError, TypeError) as e:
                logger.error(
                    f"Knowledge graph file corrupted or invalid format: {e}. "
                    f"Re-run indexing pipeline to regenerate."
                )
                self.config.enable_knowledge_graph = False
                self.degraded_components.append({
                    "component": "knowledge_graph",
                    "error": f"Invalid file format: {e}"
                })
            except Exception as e:
                # Catch-all for truly unexpected errors
                logger.critical(
                    f"Unexpected error loading knowledge graph: {type(e).__name__}: {e}",
                    exc_info=True
                )
                self.config.enable_knowledge_graph = False
                self.degraded_components.append({
                    "component": "knowledge_graph",
                    "error": f"Unexpected error: {type(e).__name__}"
                })

        # Warn if running in degraded mode
        if self.degraded_components:
            component_names = [d["component"] for d in self.degraded_components]
            logger.warning(
                f"⚠️ RUNNING IN DEGRADED MODE - Missing components: {', '.join(component_names)}"
            )
            logger.warning("Some agent tools may be unavailable or produce lower-quality results.")

        # Initialize context assembler
        logger.info("Initializing context assembler...")
        citation_format_map = {
            "inline": CitationFormat.INLINE,
            "detailed": CitationFormat.DETAILED,
            "footnote": CitationFormat.FOOTNOTE,
        }
        context_assembler = ContextAssembler(
            citation_format=citation_format_map.get(
                self.config.cli_config.citation_format, CitationFormat.INLINE
            )
        )

        # Initialize tools (CRITICAL - must happen before AgentCore)
        logger.info("Initializing tools...")
        registry = get_registry()
        registry.initialize_tools(
            vector_store=vector_store,
            embedder=embedder,
            reranker=reranker,
            graph_retriever=graph_retriever,
            knowledge_graph=knowledge_graph,
            context_assembler=context_assembler,
            config=self.config.tool_config,
        )
        logger.info(f"✅ {len(registry)} tools initialized")

        # Initialize AgentCore (reuses existing implementation)
        self.agent = AgentCore(self.config)

        # Initialize with document list
        self.agent.initialize_with_documents()

        logger.info(
            f"AgentAdapter initialized: model={self.config.model}, "
            f"vector_store={self.config.vector_store_path}, tools={len(registry)}"
        )

    async def stream_response(
        self,
        query: str,
        conversation_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream agent response as SSE-compatible events.

        Yields SSE events in format:
        {
            "event": "text_delta" | "tool_call" | "tool_calls_summary" | "cost_update" | "done" | "error",
            "data": {...}
        }

        Event types:
        - text_delta: Streaming text chunks from agent response
        - tool_call: Tool invocation detected (streamed immediately when Claude decides to use tool)
        - tool_calls_summary: Summary of all tool calls with results (sent after response completes)
        - cost_update: Token usage and cost information
        - done: Stream completed successfully
        - error: Error occurred during streaming

        Args:
            query: User query
            conversation_id: Optional conversation ID for context

        Yields:
            Dict containing event type and data
        """
        # Reset cost tracker for this request
        reset_global_tracker()
        tracker = get_global_tracker()

        try:
            # Stream from AgentCore using process_message
            # Note: process_message is synchronous generator, but we can iterate
            # in async context with periodic yields to event loop

            # Get streaming generator from AgentCore
            text_stream = self.agent.process_message(query, stream=True)

            # Stream text chunks and detect tool calls
            for chunk in text_stream:
                # Debug: Log chunk content (first 100 chars)
                logger.debug(f"Chunk received: {repr(chunk[:100])}")

                # Detect tool call notification: [Using TOOL_NAME...]
                # Pattern: [Using <tool_name>...]
                tool_call_match = re.search(r'\[Using\s+([a-z_]+)\.{3}\]', chunk)

                if tool_call_match:
                    # Extract tool name
                    tool_name = tool_call_match.group(1)
                    logger.info(f"🔧 Tool call detected: {tool_name}")

                    # Send tool_call event immediately
                    yield {
                        "event": "tool_call",
                        "data": {
                            "tool_name": tool_name,
                            "tool_input": {},  # Input not available yet (streamed before execution)
                            "call_id": f"tool_{tool_name}"  # Placeholder ID
                        }
                    }
                    # Yield control to event loop
                    await asyncio.sleep(0)
                else:
                    # Regular text content - strip ANSI color codes
                    clean_chunk = re.sub(r'\033\[[0-9;]+m', '', chunk)

                    if clean_chunk:  # Only send non-empty chunks
                        yield {
                            "event": "text_delta",
                            "data": {
                                "content": clean_chunk
                            }
                        }
                        # Yield control to event loop (allows concurrent requests)
                        await asyncio.sleep(0)

            # Extract tool calls from conversation history
            # Tool calls are stored in assistant message content blocks (Anthropic format)
            # Tool results are in subsequent user messages with metadata
            #
            # Three-pass extraction is required because:
            # 1. Tool calls (tool_use) are in assistant messages with role="assistant"
            # 2. Tool results (tool_result) are in user messages with role="user"
            # 3. They must be joined by tool_use_id to create complete tool call objects
            #
            # Performance optimization: Scan last 10 messages only (not entire history)
            # Rationale: In tool-heavy conversations, each turn can generate 2-5 messages:
            #   - 1 assistant message (with tool_use blocks)
            #   - 1-4 user messages (one tool_result per tool called)
            # Therefore, 10 messages covers the most recent 2-4 turns, which is sufficient
            # because tool_use/tool_result pairs are always in adjacent messages.
            # For no-tool conversations: 10 messages = 5 complete turns.
            tool_calls_info = []
            tool_results_map = {}  # tool_use_id -> result metadata

            if self.agent.conversation_history:
                # Pass 1: Collect tool_use blocks (from assistant messages)
                for message in self.agent.conversation_history[-10:]:
                    if message.get("role") == "assistant" and "content" in message:
                        content = message["content"]

                        # Validate content is a list (defensive programming)
                        if not isinstance(content, list):
                            logger.error(
                                f"Invalid message content format: expected list, got {type(content).__name__}. "
                                f"Message role={message.get('role')}, content preview={str(content)[:100]}"
                            )
                            continue

                        for content_block in content:
                            # Validate content_block is a dict
                            if not isinstance(content_block, dict):
                                logger.warning(f"Skipping non-dict content block: {type(content_block).__name__}")
                                continue

                            if content_block.get("type") == "tool_use":
                                # Validate required fields exist
                                if "id" not in content_block or "name" not in content_block:
                                    logger.error(
                                        f"tool_use block missing required fields. "
                                        f"Has id={('id' in content_block)}, name={('name' in content_block)}"
                                    )
                                    continue

                                tool_calls_info.append({
                                    "id": content_block.get("id", ""),
                                    "name": content_block.get("name", ""),
                                    "input": content_block.get("input", {}),
                                })

                # Pass 2: Collect tool_result blocks with metadata (from user messages)
                for message in self.agent.conversation_history[-10:]:
                    if message.get("role") == "user" and "content" in message:
                        content = message["content"]

                        # Validate content is a list
                        if not isinstance(content, list):
                            logger.error(
                                f"Invalid message content format: expected list, got {type(content).__name__}. "
                                f"Message role={message.get('role')}"
                            )
                            continue

                        for content_block in content:
                            # Validate content_block is a dict
                            if not isinstance(content_block, dict):
                                logger.warning(f"Skipping non-dict content block: {type(content_block).__name__}")
                                continue

                            if content_block.get("type") == "tool_result":
                                tool_use_id = content_block.get("tool_use_id")
                                if tool_use_id:
                                    # Note: _metadata was removed from agent_core.py (API compliance fix)
                                    # We can no longer access execution_time_ms, success, error from here
                                    # This will be handled differently in future versions
                                    tool_results_map[tool_use_id] = {
                                        "result": content_block.get("content"),
                                        "metadata": {},  # Empty - no longer available from API responses
                                    }

                # Pass 3: Merge tool_use and tool_result data by tool_use_id
                for tool_call in tool_calls_info:
                    tool_id = tool_call["id"]
                    if tool_id in tool_results_map:
                        result_data = tool_results_map[tool_id]
                        tool_call["result"] = result_data.get("result")

                        # Defensive: Get metadata with fallback to empty dict
                        metadata = result_data.get("metadata", {})
                        tool_call["executionTimeMs"] = metadata.get("execution_time_ms", 0)
                        tool_call["success"] = metadata.get("success", True)
                        tool_call["error"] = metadata.get("error")
                        tool_call["explicitParams"] = metadata.get("explicit_params", [])

            # Send tool calls summary if any
            if tool_calls_info:
                logger.info(f"Extracted {len(tool_calls_info)} tool calls from conversation history")
                yield {
                    "event": "tool_calls_summary",
                    "data": {
                        "tool_calls": tool_calls_info,
                        "count": len(tool_calls_info)
                    }
                }
            else:
                logger.debug("No tool calls found in conversation history")

            # Send final cost update
            cost_summary = tracker.get_session_cost_summary()
            yield {
                "event": "cost_update",
                "data": {
                    "summary": cost_summary,
                    "total_cost": tracker.get_total_cost(),
                    "input_tokens": tracker._total_input_tokens,
                    "output_tokens": tracker._total_output_tokens,
                    "cached_tokens": tracker._total_cache_read_tokens
                }
            }

            # Signal completion
            yield {
                "event": "done",
                "data": {}
            }

        except Exception as e:
            logger.error(f"Error during streaming: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": {
                    "error": str(e),
                    "type": type(e).__name__
                }
            }

    def get_available_models(self) -> list[Dict[str, Any]]:
        """
        Get list of available models (same as CLI).

        Returns:
            List of model info dicts matching CLI's _list_available_models()
        """
        return [
            # Anthropic Claude models (from CLI)
            {
                "id": "claude-haiku-4-5-20251001",
                "name": "Claude Haiku 4.5",
                "provider": "anthropic",
                "description": "Fast & cost-effective (✅ caching)"
            },
            {
                "id": "claude-sonnet-4-5-20250929",
                "name": "Claude Sonnet 4.5",
                "provider": "anthropic",
                "description": "Balanced performance (✅ caching)"
            },
            # Google Gemini models
            {
                "id": "gemini-2.5-flash",
                "name": "Gemini 2.5 Flash",
                "provider": "google",
                "description": "Fast & agentic (✅ caching, 250/day free)"
            },
            {
                "id": "gemini-2.5-pro",
                "name": "Gemini 2.5 Pro",
                "provider": "google",
                "description": "Best reasoning (✅ caching, 100/day free)"
            },
            {
                "id": "gemini-2.5-flash-lite",
                "name": "Gemini 2.5 Flash Lite",
                "provider": "google",
                "description": "High volume (✅ caching, 1000/day free)"
            },
            # OpenAI GPT-5 models (from CLI)
            {
                "id": "gpt-5-nano",
                "name": "GPT-5 Nano",
                "provider": "openai",
                "description": "Ultra-fast, minimal cost (❌ no caching)"
            },
            {
                "id": "gpt-5-mini",
                "name": "GPT-5 Mini",
                "provider": "openai",
                "description": "Balanced & affordable (❌ no caching)"
            },
            {
                "id": "gpt-5",
                "name": "GPT-5",
                "provider": "openai",
                "description": "Most capable (❌ no caching)"
            }
        ]

    def switch_model(self, model: str) -> None:
        """
        Switch to a different model.

        Args:
            model: Model identifier
        """
        self.config.model = model
        self.agent = AgentCore(self.config)
        logger.info(f"Switched to model: {model}")

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get agent health status.

        Returns:
            Health status dict with degraded component warnings
        """
        try:
            # Check if agent is properly initialized
            if not self.agent:
                return {
                    "status": "error",
                    "message": "Agent not initialized",
                    "details": {},
                    "degraded_components": []
                }

            # Check vector store
            vector_store_exists = self.config.vector_store_path.exists()

            # Check API keys
            has_anthropic_key = bool(self.config.anthropic_api_key)
            has_openai_key = bool(self.config.openai_api_key)
            has_google_key = bool(self.config.google_api_key)

            if not vector_store_exists:
                return {
                    "status": "error",
                    "message": "Vector store not found",
                    "details": {
                        "vector_store_path": str(self.config.vector_store_path)
                    },
                    "degraded_components": []
                }

            if not has_anthropic_key and not has_openai_key and not has_google_key:
                return {
                    "status": "error",
                    "message": "No API keys configured",
                    "details": {},
                    "degraded_components": []
                }

            # Determine overall status based on degraded components
            status = "degraded" if self.degraded_components else "healthy"
            message = "Agent ready" if status == "healthy" else "Agent running in degraded mode"

            return {
                "status": status,
                "message": message,
                "details": {
                    "model": self.config.model,
                    "vector_store": str(self.config.vector_store_path),
                    "has_anthropic_key": has_anthropic_key,
                    "has_openai_key": has_openai_key,
                    "has_google_key": has_google_key
                },
                "degraded_components": self.degraded_components  # Expose to UI
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e),
                "details": {}
            }
