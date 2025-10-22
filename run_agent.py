#!/usr/bin/env python
"""
RAG Agent CLI - Entry Point

Interactive document assistant powered by Claude SDK.

Usage:
    # Basic usage
    python run_agent.py --store output/hybrid_store

    # With knowledge graph
    python run_agent.py --store output/hybrid_store --kg output/knowledge_graph.json

    # Custom model
    python run_agent.py --store output/hybrid_store --model claude-sonnet-4-5-20250929

    # Enable HyDE
    python run_agent.py --store output/hybrid_store --enable-hyde

    # Non-streaming mode
    python run_agent.py --store output/hybrid_store --no-stream

Environment variables:
    ANTHROPIC_API_KEY   - Required
    AGENT_MODEL         - Override model selection
    VECTOR_STORE_PATH   - Default vector store path
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from src.agent.cli import main as cli_main
from src.agent.config import AgentConfig


def setup_logging(debug: bool = False, verbose: bool = False):
    """
    Setup logging configuration.

    Args:
        debug: Enable DEBUG level logging to file and console
        verbose: Enable INFO level logging to console (superseded by debug)
    """
    # Determine logging level
    if debug:
        level = logging.DEBUG
        console_level = logging.DEBUG
    elif verbose:
        level = logging.INFO
        console_level = logging.INFO
    else:
        level = logging.WARNING
        console_level = logging.ERROR

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Format strings
    detailed_format = "%(asctime)s | %(name)-30s | %(levelname)-8s | %(funcName)-20s | %(message)s"
    simple_format = "%(levelname)s: %(message)s"

    # File handler - always log to file with detailed format
    file_handler = logging.FileHandler("agent.log", mode="w")  # Overwrite each run
    file_handler.setLevel(logging.DEBUG)  # Always debug level in file
    file_handler.setFormatter(logging.Formatter(detailed_format))

    # Console handler - conditional based on debug/verbose
    if debug or verbose:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(console_level)

        # Use detailed format for debug, simple for verbose
        if debug:
            console_handler.setFormatter(logging.Formatter(detailed_format))
        else:
            console_handler.setFormatter(logging.Formatter(simple_format))

        root_logger.addHandler(console_handler)

    root_logger.addHandler(file_handler)
    root_logger.setLevel(level)

    # Log startup info
    if debug:
        logging.info("=" * 80)
        logging.info("RAG AGENT DEBUG MODE ENABLED")
        logging.info("=" * 80)
        logging.debug(f"Python version: {sys.version}")
        logging.debug(f"Working directory: {os.getcwd()}")
        logging.debug(f"Logging level: {logging.getLevelName(level)}")

    return root_logger


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RAG Agent CLI - Interactive document assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--store",
        type=Path,
        help="Path to hybrid vector store directory (required)",
    )

    # Optional arguments
    parser.add_argument(
        "--kg",
        "--knowledge-graph",
        type=Path,
        dest="knowledge_graph",
        help="Path to knowledge graph JSON file (optional)",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Claude model to use (default: claude-sonnet-4-5-20250929)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens in response (default: 4096)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Model temperature (default: 0.3)",
    )

    # Feature flags
    parser.add_argument(
        "--enable-hyde",
        action="store_true",
        help="Enable HyDE (Hypothetical Document Embeddings)",
    )

    parser.add_argument(
        "--enable-decomposition",
        action="store_true",
        help="Enable query decomposition",
    )

    parser.add_argument(
        "--no-reranking",
        action="store_true",
        help="Disable cross-encoder reranking",
    )

    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming responses",
    )

    # Display options
    parser.add_argument(
        "--citation-format",
        choices=["inline", "detailed", "footnote"],
        default="inline",
        help="Citation format (default: inline)",
    )

    parser.add_argument(
        "--hide-tool-calls",
        action="store_true",
        help="Don't show tool execution messages",
    )

    # Debug options
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (INFO level)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with comprehensive logging (DEBUG level)",
    )

    args = parser.parse_args()

    # Validate required arguments
    if not args.store:
        # Try to use default from environment
        default_store = os.getenv("VECTOR_STORE_PATH", "output/hybrid_store")
        args.store = Path(default_store)
        print(f"Using default vector store: {args.store}")

    return args


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()

    # Setup logging (debug takes precedence over verbose)
    logger = setup_logging(debug=args.debug, verbose=args.verbose)

    if args.debug:
        logger.info("Starting RAG Agent in DEBUG mode")
        logger.debug(f"Command line arguments: {vars(args)}")

    # Create config
    try:
        if args.debug:
            logger.debug("Creating AgentConfig from environment and arguments")

        config = AgentConfig.from_env(
            vector_store_path=args.store,
            knowledge_graph_path=args.knowledge_graph,
            enable_knowledge_graph=(args.knowledge_graph is not None),
            model=args.model if args.model else None,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            enable_hyde=args.enable_hyde,
            enable_query_decomposition=args.enable_decomposition,
        )

        if args.debug:
            logger.debug(f"Vector store path: {config.vector_store_path}")
            logger.debug(f"Knowledge graph: {config.enable_knowledge_graph}")
            logger.debug(f"Model: {config.model}")
            logger.debug(f"HyDE enabled: {config.enable_hyde}")
            logger.debug(f"Query decomposition enabled: {config.enable_query_decomposition}")

        # Override CLI config
        config.cli_config.enable_streaming = not args.no_stream
        config.cli_config.citation_format = args.citation_format
        config.cli_config.show_tool_calls = not args.hide_tool_calls

        # Override tool config
        config.tool_config.enable_reranking = not args.no_reranking

        # Store debug flag in config for component access
        config.debug_mode = args.debug

        if args.debug:
            logger.info("✅ Configuration created successfully")

    except Exception as e:
        logger.error(f"Configuration error: {e}", exc_info=True)
        print(f"❌ Configuration error: {e}")
        if args.debug:
            print(f"\n📝 See agent.log for detailed error trace")
        sys.exit(1)

    # Run CLI
    try:
        if args.debug:
            logger.info("Starting CLI main loop")

        cli_main(config)

    except KeyboardInterrupt:
        if args.debug:
            logger.info("Received keyboard interrupt")
        print("\n\n👋 Interrupted. Goodbye!")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        if args.debug:
            print(f"\n📝 See agent.log for detailed error trace")
        sys.exit(1)


if __name__ == "__main__":
    main()
