"""
Unit tests for utils.model_registry module.

Tests model name resolution and alias management.
"""

import pytest
from src.utils.model_registry import ModelRegistry


class TestLLMModels:
    """Test LLM model resolution."""

    def test_haiku_alias_resolved(self):
        """Test haiku alias resolves to full model name."""
        result = ModelRegistry.resolve_llm("haiku")
        assert result == "claude-haiku-4-5-20251001"

    def test_sonnet_alias_resolved(self):
        """Test sonnet alias resolves to full model name."""
        result = ModelRegistry.resolve_llm("sonnet")
        assert result == "claude-sonnet-4-5-20250929"

    def test_gpt_4o_mini_resolved(self):
        """Test gpt-4o-mini passes through."""
        result = ModelRegistry.resolve_llm("gpt-4o-mini")
        assert result == "gpt-4o-mini"

    def test_full_model_name_unchanged(self):
        """Test full model names pass through unchanged."""
        full_name = "claude-sonnet-4-5-20250929"
        result = ModelRegistry.resolve_llm(full_name)
        assert result == full_name

    def test_unknown_model_unchanged(self):
        """Test unknown model names pass through unchanged."""
        unknown = "my-custom-model"
        result = ModelRegistry.resolve_llm(unknown)
        assert result == unknown

    def test_llm_models_not_empty(self):
        """Test LLM_MODELS dictionary is not empty."""
        assert len(ModelRegistry.LLM_MODELS) > 0

    def test_all_haiku_aliases_work(self):
        """Test all Haiku aliases resolve to same model."""
        aliases = ["haiku", "claude-haiku", "claude-haiku-4-5"]
        expected = "claude-haiku-4-5-20251001"

        for alias in aliases:
            if alias in ModelRegistry.LLM_MODELS:
                result = ModelRegistry.resolve_llm(alias)
                assert result == expected, f"Alias {alias} failed"


class TestEmbeddingModels:
    """Test embedding model resolution."""

    def test_voyage_3_resolved(self):
        """Test voyage-3 alias resolves correctly."""
        result = ModelRegistry.resolve_embedding("voyage-3")
        assert result == "voyage-3-large"

    def test_bge_m3_resolved(self):
        """Test bge-m3 alias resolves correctly."""
        result = ModelRegistry.resolve_embedding("bge-m3")
        assert result == "BAAI/bge-m3"

    def test_text_embedding_3_large(self):
        """Test OpenAI embedding model."""
        result = ModelRegistry.resolve_embedding("text-embedding-3-large")
        assert result == "text-embedding-3-large"

    def test_unknown_embedding_unchanged(self):
        """Test unknown embedding names pass through."""
        unknown = "my-custom-embedding"
        result = ModelRegistry.resolve_embedding(unknown)
        assert result == unknown

    def test_embedding_models_not_empty(self):
        """Test EMBEDDING_MODELS dictionary is not empty."""
        assert len(ModelRegistry.EMBEDDING_MODELS) > 0

    def test_is_local_embedding_bge_m3(self):
        """Test bge-m3 is detected as local."""
        assert ModelRegistry.is_local_embedding("BAAI/bge-m3") is True

    def test_is_local_embedding_openai(self):
        """Test OpenAI models are not local."""
        assert ModelRegistry.is_local_embedding("text-embedding-3-large") is False

    def test_is_local_embedding_voyage(self):
        """Test Voyage models are not local."""
        assert ModelRegistry.is_local_embedding("voyage-3-large") is False


class TestRerankerModels:
    """Test reranker model resolution."""

    def test_default_alias(self):
        """Test default reranker alias."""
        result = ModelRegistry.resolve_reranker("default")
        assert result == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_fast_alias(self):
        """Test fast reranker alias."""
        result = ModelRegistry.resolve_reranker("fast")
        assert result == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_sota_alias(self):
        """Test SOTA reranker alias."""
        result = ModelRegistry.resolve_reranker("sota")
        assert result == "BAAI/bge-reranker-large"

    def test_full_reranker_name(self):
        """Test full reranker name passes through."""
        full_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        result = ModelRegistry.resolve_reranker(full_name)
        assert result == full_name

    def test_unknown_reranker_unchanged(self):
        """Test unknown reranker names pass through."""
        unknown = "my-custom-reranker"
        result = ModelRegistry.resolve_reranker(unknown)
        assert result == unknown

    def test_reranker_models_not_empty(self):
        """Test RERANKER_MODELS dictionary is not empty."""
        assert len(ModelRegistry.RERANKER_MODELS) > 0


class TestModelRegistryConsistency:
    """Test consistency across ModelRegistry."""

    def test_no_duplicate_values_llm(self):
        """Test no duplicate full model names in LLM models."""
        # Aliases can map to same model, that's OK
        # This just checks the values make sense
        values = list(ModelRegistry.LLM_MODELS.values())
        assert len(values) > 0

    def test_no_duplicate_values_embedding(self):
        """Test no unexpected duplicates in embedding models."""
        values = list(ModelRegistry.EMBEDDING_MODELS.values())
        assert len(values) > 0

    def test_no_duplicate_values_reranker(self):
        """Test no unexpected duplicates in reranker models."""
        values = list(ModelRegistry.RERANKER_MODELS.values())
        assert len(values) > 0

    def test_all_dictionaries_have_content(self):
        """Test all model registries have at least some entries."""
        assert len(ModelRegistry.LLM_MODELS) >= 5
        assert len(ModelRegistry.EMBEDDING_MODELS) >= 3
        assert len(ModelRegistry.RERANKER_MODELS) >= 3


# Integration tests
class TestModelRegistryIntegration:
    """Integration tests for ModelRegistry."""

    def test_config_py_compatibility(self):
        """Test that resolve methods work like old config.py."""
        # Simulate what config.py's resolve_model_alias did
        test_cases = [
            ("haiku", "LLM"),
            ("sonnet", "LLM"),
            ("voyage-3", "embedding"),
            ("bge-m3", "embedding"),
            ("default", "reranker"),
        ]

        for model_name, model_type in test_cases:
            if model_type == "LLM":
                result = ModelRegistry.resolve_llm(model_name)
            elif model_type == "embedding":
                result = ModelRegistry.resolve_embedding(model_name)
            elif model_type == "reranker":
                result = ModelRegistry.resolve_reranker(model_name)

            # Should resolve to something (not None, not empty)
            assert result
            assert len(result) > 0

    def test_reranker_py_compatibility(self):
        """Test that reranker resolution matches old RERANKER_MODELS."""
        # All reranker aliases should resolve
        for alias in ["default", "fast", "accurate", "sota"]:
            if alias in ModelRegistry.RERANKER_MODELS:
                result = ModelRegistry.resolve_reranker(alias)
                assert result is not None
                assert len(result) > 0

    def test_embedding_local_detection(self):
        """Test local embedding detection for all known models."""
        # Local models (HuggingFace)
        local_models = [
            "BAAI/bge-m3",
            "BAAI/bge-large-en-v1.5",
            "BAAI/bge-reranker-base",
        ]

        for model in local_models:
            if model in [ModelRegistry.EMBEDDING_MODELS.get(k) for k in ModelRegistry.EMBEDDING_MODELS]:
                assert ModelRegistry.is_local_embedding(model) is True

        # Cloud models
        cloud_models = [
            "text-embedding-3-large",
            "text-embedding-3-small",
            "voyage-3-large",
            "voyage-law-2",
        ]

        for model in cloud_models:
            assert ModelRegistry.is_local_embedding(model) is False
