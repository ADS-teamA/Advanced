"""
Entry point for agent CLI when invoked as: python -m src.agent.cli
"""

import sys
import argparse
from pathlib import Path

# Use absolute imports for -m invocation
from src.agent.config import AgentConfig, CLIConfig
from src.agent.cli import main


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
        print(f"📝 Args parsed: vector_store={args.vector_store}")

        # Create config from args
        config = AgentConfig(
            vector_store_path=Path(args.vector_store),
            model=args.model,
            debug_mode=args.debug,
            cli_config=CLIConfig(
                enable_streaming=not args.no_streaming
            )
        )
        print(f"✅ Config created")

        # Run main
        print(f"🔧 Calling main()...")
        main(config)

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
