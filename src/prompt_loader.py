"""
Prompt Loader Utility

Centralized utility for loading LLM prompts from the prompts/ directory.
All prompts are now stored as text files for easy editing and maintenance.
"""

import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"


def load_prompt(prompt_name: str) -> str:
    """
    Load a prompt from the prompts/ directory.

    Args:
        prompt_name: Name of the prompt file (with or without .txt extension)

    Returns:
        Prompt text as string

    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    # Add .txt extension if not present
    if not prompt_name.endswith('.txt'):
        prompt_name = f"{prompt_name}.txt"

    prompt_path = PROMPTS_DIR / prompt_name

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}\n"
            f"Expected location: {PROMPTS_DIR}\n"
            f"Available prompts: {list(PROMPTS_DIR.glob('*.txt'))}"
        )

    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read()

        logger.debug(f"Loaded prompt: {prompt_name} ({len(prompt)} chars)")
        return prompt

    except Exception as e:
        logger.error(f"Failed to load prompt {prompt_name}: {e}")
        raise


def load_prompts(*prompt_names: str) -> Dict[str, str]:
    """
    Load multiple prompts at once.

    Args:
        *prompt_names: Prompt file names to load

    Returns:
        Dictionary mapping prompt names to prompt text
    """
    prompts = {}
    for name in prompt_names:
        prompts[name] = load_prompt(name)
    return prompts


# Pre-load commonly used prompts for performance
_PROMPT_CACHE = {}


def get_prompt(prompt_name: str, use_cache: bool = True) -> str:
    """
    Get a prompt with optional caching.

    Args:
        prompt_name: Name of the prompt file
        use_cache: Whether to use cached version (default: True)

    Returns:
        Prompt text
    """
    if use_cache and prompt_name in _PROMPT_CACHE:
        return _PROMPT_CACHE[prompt_name]

    prompt = load_prompt(prompt_name)

    if use_cache:
        _PROMPT_CACHE[prompt_name] = prompt

    return prompt
