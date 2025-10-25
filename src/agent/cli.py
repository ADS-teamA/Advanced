"""
CLI REPL Interface

Interactive terminal interface for RAG Agent with:
- Startup validation
- REPL loop
- Commands (/help, /stats, /config, /exit)
- Streaming display
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from src.context_assembly import CitationFormat, ContextAssembler
from src.embedding_generator import EmbeddingConfig, EmbeddingGenerator

# Import pipeline components
from src.hybrid_search import HybridVectorStore
from src.reranker import CrossEncoderReranker

from .agent_core import AgentCore
from .config import AgentConfig
from .tools.registry import get_registry

logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
COLOR_GREEN = "\033[1;32m"  # Bold green for assistant messages
COLOR_RESET = "\033[0m"  # Reset color


class AgentCLI:
    """
    CLI interface for RAG Agent.

    Handles:
    - Startup validation (check vector store)
    - Component initialization
    - REPL loop
    - Command handling
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize CLI.

        Args:
            config: AgentConfig instance
        """
        self.config = config
        self.agent: Optional[AgentCore] = None

    def startup_validation(self) -> bool:
        """
        Validate environment before starting agent.

        Uses comprehensive validator to check:
        - Python version compatibility
        - Required dependencies
        - API keys
        - Vector store integrity
        - Knowledge graph (if enabled)
        - Component integration

        Returns:
            True if valid, False otherwise
        """
        from src.agent.validation import AgentValidator

        logger.info("Starting comprehensive validation...")

        # Create validator
        validator = AgentValidator(self.config, debug=self.config.debug_mode)

        # Run all checks
        try:
            validation_passed = validator.validate_all()

            # Print summary to console
            if not self.config.debug_mode:
                # In normal mode, print condensed summary
                validator.print_summary()
            else:
                # In debug mode, detailed log already printed
                print("\n📝 Detailed validation log written to agent.log")

            if not validation_passed:
                print("\n❌ Validation failed. Please fix errors above before continuing.")
                return False

            print("\n✅ All validation checks passed - agent ready to start\n")
            return True

        except Exception as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            print(f"\n❌ Validation error: {e}")
            if self.config.debug_mode:
                print(f"📝 See agent.log for detailed error trace")
            return False

    def initialize_agent(self):
        """
        Initialize agent and all pipeline components.

        Loads:
        - Vector store
        - Embedder
        - Reranker (optional)
        - Knowledge graph (optional)
        - Context assembler
        """
        print("🚀 Initializing agent components...")

        # Load vector store with error handling
        print("Loading vector store...")
        try:
            vector_store = HybridVectorStore.load(self.config.vector_store_path)
        except FileNotFoundError:
            logger.error(f"Vector store not found: {self.config.vector_store_path}")
            raise RuntimeError(
                f"❌ Vector store not found at: {self.config.vector_store_path}\n"
                f"   Please run the indexing pipeline first:\n"
                f"   python run_pipeline.py data/your_documents/"
            )
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}", exc_info=True)
            raise RuntimeError(
                f"❌ Vector store loading failed: {e}\n"
                f"   The vector store may be corrupted. Try re-indexing:\n"
                f"   python run_pipeline.py data/your_documents/"
            )

        # Initialize embedder with error handling (platform-aware model selection)
        print(f"Initializing embedder (model: {self.config.embedding_model})...")
        try:
            embedder = EmbeddingGenerator(
                EmbeddingConfig(model=self.config.embedding_model, batch_size=100, normalize=True)
            )
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {e}", exc_info=True)
            # Check if it's an API key issue
            if "api" in str(e).lower() or "key" in str(e).lower():
                raise RuntimeError(
                    f"❌ Embedder initialization failed: {e}\n"
                    f"   Model: {self.config.embedding_model}\n"
                    f"   This model requires an API key. Please set:\n"
                    f"   export OPENAI_API_KEY=your_key_here  # For OpenAI models\n"
                    f"   Or use a local model by setting:\n"
                    f"   export EMBEDDING_MODEL=bge-m3"
                )
            raise RuntimeError(
                f"❌ Embedder initialization failed: {e}\n"
                f"   Model: {self.config.embedding_model}\n"
                f"   Check that the model is supported and dependencies are installed."
            )

        # Initialize reranker (optional, lazy load)
        reranker = None
        if self.config.tool_config.enable_reranking:
            if not self.config.tool_config.lazy_load_reranker:
                print("Loading reranker...")
                try:
                    reranker = CrossEncoderReranker(
                        model_name=self.config.tool_config.reranker_model
                    )
                except Exception as e:
                    logger.warning(f"Failed to load reranker: {e}. Continuing without reranking.")
                    print(f"   ⚠️  Reranker failed to load: {e}")
                    print("   Continuing without reranking (results may be less accurate)")
                    self.config.tool_config.enable_reranking = False
            else:
                print("Reranker set to lazy load")

        # Load knowledge graph (optional)
        knowledge_graph = None
        graph_retriever = None
        if self.config.enable_knowledge_graph and self.config.knowledge_graph_path:
            print("Loading knowledge graph...")
            try:
                from src.graph.models import KnowledgeGraph
                from src.graph_retrieval import GraphEnhancedRetriever

                knowledge_graph = KnowledgeGraph.load_json(str(self.config.knowledge_graph_path))
                print(
                    f"   Entities: {len(knowledge_graph.entities)}, "
                    f"Relationships: {len(knowledge_graph.relationships)}"
                )

                graph_retriever = GraphEnhancedRetriever(
                    vector_store=vector_store, knowledge_graph=knowledge_graph
                )
            except FileNotFoundError:
                logger.warning(
                    f"Knowledge graph file not found: {self.config.knowledge_graph_path}"
                )
                print(f"   ⚠️  Knowledge graph file not found: {self.config.knowledge_graph_path}")
                print("   Continuing without knowledge graph (graph tools will be unavailable)")
                self.config.enable_knowledge_graph = False
            except ImportError as e:
                logger.warning(f"Knowledge graph module not available: {e}")
                print(f"   ⚠️  Knowledge graph module not available: {e}")
                print("   Continuing without knowledge graph")
                self.config.enable_knowledge_graph = False
            except Exception as e:
                logger.warning(f"Failed to load knowledge graph: {e}")
                print(f"   ⚠️  Knowledge graph failed to load: {e}")
                print("   Continuing without knowledge graph")
                self.config.enable_knowledge_graph = False

        # Initialize context assembler
        print("Initializing context assembler...")
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

        # Initialize tools
        print("Initializing tools...")
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

        print(f"✅ {len(registry)} tools initialized\n")

        # Create agent
        self.agent = AgentCore(self.config)

        # Initialize with document list (adds to conversation history)
        self.agent.initialize_with_documents()

        # Check for degraded mode
        degraded_features = []
        if self.config.enable_knowledge_graph and knowledge_graph is None:
            degraded_features.append("Knowledge Graph (graph tools unavailable)")
        if self.config.tool_config.enable_reranking and reranker is None:
            degraded_features.append("Reranking (search quality reduced)")

        if degraded_features:
            print("⚠️  DEGRADED MODE ACTIVE:")
            for feature in degraded_features:
                print(f"   • {feature}")
            print("\nAgent will run with limited functionality.")
            print("To enable missing features, check configuration and dependencies.\n")

        print("✅ Agent ready!\n")

    def run_repl(self):
        """
        Run interactive REPL loop.

        Commands:
        - /help: Show help
        - /stats: Show tool statistics
        - /config: Show configuration
        - /clear: Clear conversation
        - /exit or /quit: Exit
        """
        print("╭─────────────────────────────────────────────────────────────╮")
        print("│              RAG Agent - Document Assistant                 │")
        print("│  Type your question or use /help for commands              │")
        print("╰─────────────────────────────────────────────────────────────╯\n")

        while True:
            try:
                # Get user input
                user_input = input("\n> ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                    continue

                # Process message with agent
                if self.config.cli_config.enable_streaming:
                    print(f"\n{COLOR_GREEN}A: {COLOR_RESET}", end="", flush=True)
                    for chunk in self.agent.process_message(user_input, stream=True):
                        # Check if chunk starts with ANSI color code (tool call/debug)
                        if chunk.startswith("\033["):
                            # Don't colorize - already has color
                            print(chunk, end="", flush=True)
                        else:
                            # Colorize assistant message in green
                            print(f"{COLOR_GREEN}{chunk}{COLOR_RESET}", end="", flush=True)
                    print()  # Newline after response
                else:
                    response = self.agent.process_message(user_input, stream=False)
                    print(f"\n{COLOR_GREEN}A: {response}{COLOR_RESET}")

                # Show session cost after each response
                cost_summary = self.agent.tracker.get_session_cost_summary()
                print(f"\n{cost_summary}")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                print(f"\n❌ Error: {e}")
                print("Try again or use /help for assistance.")

    def _handle_command(self, command: str):
        """Handle CLI commands."""
        cmd = command.lower().split()[0]

        if cmd in ["/help", "/h"]:
            self._show_help()

        elif cmd in ["/stats", "/s"]:
            self._show_stats()

        elif cmd in ["/config", "/c"]:
            self._show_config()

        elif cmd in ["/clear", "/reset"]:
            self.agent.reset_conversation()
            print("✅ Conversation cleared")

        elif cmd in ["/exit", "/quit", "/q"]:
            print("\n👋 Goodbye!")
            sys.exit(0)

        else:
            print(f"❌ Unknown command: {cmd}")
            print("Use /help to see available commands")

    def _show_help(self):
        """Show help message."""
        print("\n📖 Available Commands:")
        print("  /help, /h        - Show this help")
        print("  /stats, /s       - Show tool execution and cost statistics")
        print("  /config, /c      - Show current configuration")
        print("  /clear, /reset   - Clear conversation and reinitialize")
        print("  /exit, /quit, /q - Exit the agent")
        print("\n💡 Tips:")
        print("  - Just type your question to start")
        print("  - Agent has access to 27 specialized tools")
        print("  - Use specific questions for best results")
        print("  - Citations are included in responses")
        print("  - Session cost is shown after each response")

    def _show_stats(self):
        """Show tool execution statistics."""
        registry = get_registry()
        stats = registry.get_stats()

        print("\n📊 Tool Execution Statistics:")
        print(f"Total tools: {stats['total_tools']}")
        print(f"Total calls: {stats['total_calls']}")
        print(f"Total errors: {stats['total_errors']}")
        print(f"Success rate: {stats['success_rate']}%")
        print(f"Total time: {stats['total_time_ms']:.0f}ms")
        print(f"Avg time per call: {stats['avg_time_ms']:.0f}ms")

        # Show top 5 most used tools
        if stats["tools"]:
            sorted_tools = sorted(stats["tools"], key=lambda x: x["execution_count"], reverse=True)
            print("\n🔝 Most Used Tools:")
            for tool in sorted_tools[:5]:
                if tool["execution_count"] > 0:
                    print(
                        f"  {tool['name']:20s} - {tool['execution_count']:3d} calls, "
                        f"{tool['avg_time_ms']:6.0f}ms avg"
                    )

        # Show conversation stats
        if self.agent:
            conv_stats = self.agent.get_conversation_stats()
            print("\n💬 Conversation Statistics:")
            print(f"  Messages: {conv_stats['message_count']}")
            print(f"  Tool calls: {conv_stats['tool_calls']}")
            if conv_stats["tools_used"]:
                print(f"  Tools used: {', '.join(conv_stats['tools_used'])}")

            # Show cost statistics
            tracker = self.agent.tracker
            total_cost = tracker.get_total_cost()
            total_tokens = tracker.get_total_tokens()
            cache_stats = tracker.get_cache_stats()

            print("\n💰 Cost Statistics:")
            print(f"  Total cost: ${total_cost:.4f}")
            print(f"  Total tokens: {total_tokens:,}")
            print(f"    Input: {tracker.total_input_tokens:,}")
            print(f"    Output: {tracker.total_output_tokens:,}")

            # Show cache stats if caching was used
            if cache_stats["cache_read_tokens"] > 0 or cache_stats["cache_creation_tokens"] > 0:
                print("\n📦 Cache Statistics (Prompt Caching):")
                print(f"  Cache read: {cache_stats['cache_read_tokens']:,} tokens (90% saved)")
                print(f"  Cache created: {cache_stats['cache_creation_tokens']:,} tokens")

            # Show tool token/cost statistics
            if self.agent.tool_call_history:
                total_tool_tokens = sum(
                    call.get("estimated_tokens", 0) for call in self.agent.tool_call_history
                )
                total_tool_cost = sum(
                    call.get("estimated_cost", 0.0) for call in self.agent.tool_call_history
                )

                print("\n🔧 Tool Result Statistics:")
                print(f"  Total tool results: {len(self.agent.tool_call_history)}")
                print(f"  Total tokens (estimated): {total_tool_tokens:,}")
                print(f"  Total cost (estimated): ${total_tool_cost:.6f}")

                # Show top 5 tools by token consumption
                tool_tokens_by_name = {}
                for call in self.agent.tool_call_history:
                    tool_name = call.get("tool_name", "unknown")
                    tokens = call.get("estimated_tokens", 0)
                    cost = call.get("estimated_cost", 0.0)

                    if tool_name not in tool_tokens_by_name:
                        tool_tokens_by_name[tool_name] = {"tokens": 0, "cost": 0.0, "calls": 0}

                    tool_tokens_by_name[tool_name]["tokens"] += tokens
                    tool_tokens_by_name[tool_name]["cost"] += cost
                    tool_tokens_by_name[tool_name]["calls"] += 1

                if tool_tokens_by_name:
                    sorted_by_tokens = sorted(
                        tool_tokens_by_name.items(), key=lambda x: x[1]["tokens"], reverse=True
                    )
                    print("\n🔝 Top Tools by Token Usage:")
                    for tool_name, data in sorted_by_tokens[:5]:
                        avg_tokens = data["tokens"] / data["calls"] if data["calls"] > 0 else 0
                        print(
                            f"  {tool_name:20s} - {data['tokens']:6,} tokens "
                            f"(${data['cost']:.6f}), {data['calls']} calls, "
                            f"{avg_tokens:.0f} tokens/call"
                        )

    def _show_config(self):
        """Show current configuration."""
        print("\n⚙️  Current Configuration:")
        print(f"  Model: {self.config.model}")
        print(f"  Max tokens: {self.config.max_tokens}")
        print(f"  Temperature: {self.config.temperature}")
        print(f"  Vector store: {self.config.vector_store_path}")
        print(f"  Knowledge graph: {self.config.enable_knowledge_graph}")
        print(f"  Streaming: {self.config.cli_config.enable_streaming}")
        print(f"  Show citations: {self.config.cli_config.show_citations}")
        print(f"  Citation format: {self.config.cli_config.citation_format}")


def main(config: AgentConfig):
    """
    Main entry point for CLI.

    Args:
        config: AgentConfig instance
    """
    cli = AgentCLI(config)

    # Startup validation
    if not cli.startup_validation():
        print("\n❌ Startup validation failed. Exiting.")
        sys.exit(1)

    # Initialize agent
    try:
        cli.initialize_agent()
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}", exc_info=True)
        print(f"\n❌ Failed to initialize agent: {e}")
        sys.exit(1)

    # Run REPL
    cli.run_repl()


if __name__ == "__main__":
    """Allow running as: python -m src.agent.cli"""
    import argparse

    parser = argparse.ArgumentParser(description="RAG Agent CLI - Interactive document assistant")
    parser.add_argument(
        "--vector-store",
        type=str,
        help="Path to vector store directory",
        default=os.getenv("VECTOR_STORE_PATH", "vector_db"),
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Claude model to use",
        default=os.getenv("AGENT_MODEL", "claude-haiku-4-5"),
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-streaming", action="store_true", help="Disable streaming responses")

    args = parser.parse_args()

    from .config import CLIConfig

    config = AgentConfig(
        vector_store_path=Path(args.vector_store),
        model=args.model,
        debug_mode=args.debug,
        cli_config=CLIConfig(enable_streaming=not args.no_streaming),
    )

    try:
        main(config)
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)
