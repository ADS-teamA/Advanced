"""
Entry point for agent CLI when invoked as: python -m src.agent.cli
"""

import sys
import argparse
import logging
from pathlib import Path

# Use absolute imports for -m invocation
from src.agent.config import AgentConfig, CLIConfig
from src.agent.cli import main

logger = logging.getLogger(__name__)


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds blue color to all log messages."""

    COLOR_BLUE = "\033[1;34m"
    COLOR_RESET = "\033[0m"

    def format(self, record):
        # Format the message using parent formatter
        message = super().format(record)
        # Add blue color
        return f"{self.COLOR_BLUE}{message}{self.COLOR_RESET}"


def setup_logging(debug: bool = False):
    """
    Setup colored logging for CLI.

    Args:
        debug: Enable debug level logging
    """
    level = logging.DEBUG if debug else logging.INFO

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Use colored formatter for console output
    formatter = ColoredFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RAG Agent CLI - Interactive document assistant"
    )

    parser.add_argument(
        "--vector-store",
        type=str,
        help="Path to vector store directory",
        default="output/vector_store"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Claude model to use (default: claude-sonnet-4-5)",
        default="claude-sonnet-4-5"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming responses"
    )

    return parser.parse_args()


if __name__ == "__main__":
    import sys

    print("🚀 Starting RAG Agent CLI...")

    try:
        args = parse_args()

        # Setup colored logging
        setup_logging(debug=args.debug)

        logger.debug(f"Args parsed: vector_store={args.vector_store}, model={args.model}, debug={args.debug}")

        # Create config from args
        config = AgentConfig(
            vector_store_path=Path(args.vector_store),
            model=args.model,
            debug_mode=args.debug,
            cli_config=CLIConfig(
                enable_streaming=not args.no_streaming
            )
        )
        logger.debug("Config created successfully")

        # Run main
        logger.debug("Calling main()")
        main(config)

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
