"""
Prompt loading utilities for agent system.

Loads prompts from the prompts/ directory.
"""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_prompt(prompt_name: str) -> str:
    """
    Load a prompt from the prompts/ directory.

    Args:
        prompt_name: Name of the prompt file (without .txt extension)

    Returns:
        str: The prompt text

    Raises:
        FileNotFoundError: If prompts directory or prompt file doesn't exist
    """
    prompt_dir = Path(__file__).parent / "prompts"

    # Check if prompts directory exists
    if not prompt_dir.exists():
        raise FileNotFoundError(
            f"Prompts directory not found: {prompt_dir}\n"
            f"Expected location: src/agent/prompts/"
        )

    prompt_path = prompt_dir / f"{prompt_name}.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}\n"
            f"Expected location: src/agent/prompts/{prompt_name}.txt"
        )

    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.error(f"Failed to read prompt file {prompt_path}: {e}")
        raise
