"""
Example configurations demonstrating task-specific LLM settings.

This file shows how to configure different models for different tasks in the RAG pipeline.
You can copy and adapt these examples for your use case.
"""

from pathlib import Path
from src.config import LLMTaskConfig, LLMTasksConfig
from src.indexing_pipeline import IndexingPipeline, IndexingConfig


# ==================== EXAMPLE 1: Cost-Optimized Configuration ====================
# Use cheap models (Haiku, GPT-4o-mini) for all tasks except agent

def example_cost_optimized():
    """Minimize API costs by using cheapest models where possible."""

    llm_tasks = LLMTasksConfig(
        # Summaries: Haiku is fast and cheap
        summary=LLMTaskConfig(
            provider="claude",
            model="haiku",
            temperature=0.3,
            max_tokens=500
        ),

        # Context generation: Haiku is sufficient
        context=LLMTaskConfig(
            provider="claude",
            model="haiku",
            temperature=0.3,
            max_tokens=150
        ),

        # Entity extraction: GPT-4o-mini is cheap and accurate enough
        entity_extraction=LLMTaskConfig(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=500
        ),

        # Relationship extraction: GPT-4o-mini
        relationship_extraction=LLMTaskConfig(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=500
        ),

        # Agent: Use Sonnet for better reasoning (this is user-facing)
        agent=LLMTaskConfig(
            provider="claude",
            model="sonnet",
            temperature=0.3,
            max_tokens=4096
        ),
    )

    config = IndexingConfig(
        llm_tasks=llm_tasks,
        enable_knowledge_graph=True,
        enable_hybrid_search=True,
    )

    pipeline = IndexingPipeline(config)
    return pipeline


# ==================== EXAMPLE 2: Quality-First Configuration ====================
# Use best models for critical tasks (entity extraction, agent)

def example_quality_first():
    """Maximize quality by using best models for critical tasks."""

    llm_tasks = LLMTasksConfig(
        # Summaries: Haiku is still fine
        summary=LLMTaskConfig(
            provider="claude",
            model="haiku",
            temperature=0.3,
            max_tokens=500
        ),

        # Context: Haiku
        context=LLMTaskConfig(
            provider="claude",
            model="haiku",
            temperature=0.3,
            max_tokens=150
        ),

        # Entity extraction: GPT-4o for better accuracy
        entity_extraction=LLMTaskConfig(
            provider="openai",
            model="gpt-4o",
            temperature=0.0,
            max_tokens=500
        ),

        # Relationship extraction: GPT-4o for complex relationships
        relationship_extraction=LLMTaskConfig(
            provider="openai",
            model="gpt-4o",
            temperature=0.0,
            max_tokens=500
        ),

        # Agent: Sonnet for reasoning
        agent=LLMTaskConfig(
            provider="claude",
            model="sonnet",
            temperature=0.3,
            max_tokens=4096
        ),
    )

    config = IndexingConfig(
        llm_tasks=llm_tasks,
        enable_knowledge_graph=True,
        enable_hybrid_search=True,
        enable_reranking=True,
    )

    pipeline = IndexingPipeline(config)
    return pipeline


# ==================== EXAMPLE 3: Mixed Provider Configuration ====================
# Use Claude for some tasks, OpenAI for others

def example_mixed_providers():
    """Use different providers based on their strengths."""

    llm_tasks = LLMTasksConfig(
        # Claude is great for summaries and context
        summary=LLMTaskConfig(
            provider="claude",
            model="sonnet",  # Better quality summaries
            temperature=0.3,
            max_tokens=500
        ),

        context=LLMTaskConfig(
            provider="claude",
            model="haiku",
            temperature=0.3,
            max_tokens=150
        ),

        # OpenAI for structured extraction (entities, relationships)
        entity_extraction=LLMTaskConfig(
            provider="openai",
            model="gpt-4o",
            temperature=0.0,
            max_tokens=500
        ),

        relationship_extraction=LLMTaskConfig(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=500
        ),

        # Claude Sonnet for agent (best reasoning)
        agent=LLMTaskConfig(
            provider="claude",
            model="sonnet",
            temperature=0.3,
            max_tokens=4096
        ),
    )

    config = IndexingConfig(llm_tasks=llm_tasks)
    pipeline = IndexingPipeline(config)
    return pipeline


# ==================== EXAMPLE 4: Environment-Based Configuration ====================
# Load task-specific models from environment variables

def example_from_env():
    """
    Load configuration from environment variables.

    Set these environment variables in your .env file:
        SUMMARY_LLM_PROVIDER=claude
        SUMMARY_LLM_MODEL=haiku
        ENTITY_LLM_PROVIDER=openai
        ENTITY_LLM_MODEL=gpt-4o
        AGENT_LLM_PROVIDER=claude
        AGENT_LLM_MODEL=sonnet

    See LLMTasksConfig.from_env() docstring for all available variables.
    """

    llm_tasks = LLMTasksConfig.from_env()

    config = IndexingConfig(
        llm_tasks=llm_tasks,
        enable_knowledge_graph=True,
    )

    pipeline = IndexingPipeline(config)
    return pipeline


# ==================== EXAMPLE 5: Task-Specific Temperature Tuning ====================
# Different temperatures for different tasks

def example_temperature_tuning():
    """Fine-tune temperature for each task."""

    llm_tasks = LLMTasksConfig(
        # Low temperature for summaries (consistency)
        summary=LLMTaskConfig(
            provider="claude",
            model="haiku",
            temperature=0.2,  # More deterministic
            max_tokens=500
        ),

        # Low temperature for context (consistency)
        context=LLMTaskConfig(
            provider="claude",
            model="haiku",
            temperature=0.2,
            max_tokens=150
        ),

        # Zero temperature for extraction (deterministic)
        entity_extraction=LLMTaskConfig(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=500
        ),

        relationship_extraction=LLMTaskConfig(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=500
        ),

        # Moderate temperature for agent (creative but controlled)
        agent=LLMTaskConfig(
            provider="claude",
            model="sonnet",
            temperature=0.4,
            max_tokens=4096
        ),

        # Higher temperature for HyDE (creative hypothetical documents)
        hyde=LLMTaskConfig(
            provider="claude",
            model="haiku",
            temperature=0.7,  # More creative
            max_tokens=500
        ),
    )

    config = IndexingConfig(llm_tasks=llm_tasks)
    pipeline = IndexingPipeline(config)
    return pipeline


# ==================== EXAMPLE 6: Backward Compatible (Legacy) ====================
# Using old-style configuration (still supported)

def example_legacy_config():
    """
    Old-style configuration using legacy fields.
    This still works but is deprecated.
    """

    config = IndexingConfig(
        # Legacy fields (deprecated but still functional)
        summary_model="gpt-4o-mini",
        kg_llm_provider="openai",
        kg_llm_model="gpt-4o-mini",

        # These legacy fields will be converted to llm_tasks internally
        enable_knowledge_graph=True,
    )

    pipeline = IndexingPipeline(config)
    return pipeline


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Choose your preferred configuration
    pipeline = example_cost_optimized()
    # pipeline = example_quality_first()
    # pipeline = example_mixed_providers()
    # pipeline = example_from_env()
    # pipeline = example_temperature_tuning()

    # Index a document
    result = pipeline.index_document(Path("data/document.pdf"))

    # Access the configured models
    print("\n=== Configured Models ===")
    print(f"Summary: {pipeline.config.llm_tasks.summary.provider}/{pipeline.config.llm_tasks.summary.model}")
    print(f"Context: {pipeline.config.llm_tasks.context.provider}/{pipeline.config.llm_tasks.context.model}")
    print(f"Entity Extraction: {pipeline.config.llm_tasks.entity_extraction.provider}/{pipeline.config.llm_tasks.entity_extraction.model}")
    print(f"Relationship Extraction: {pipeline.config.llm_tasks.relationship_extraction.provider}/{pipeline.config.llm_tasks.relationship_extraction.model}")
    print(f"Agent: {pipeline.config.llm_tasks.agent.provider}/{pipeline.config.llm_tasks.agent.model}")

    # Save results
    result["vector_store"].save(Path("output/vector_store"))
    if result.get("knowledge_graph"):
        result["knowledge_graph"].save_json(Path("output/knowledge_graph.json"))

    print("\n✅ Indexing complete!")
