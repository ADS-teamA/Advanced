"""
Comprehensive Validation and Diagnostics Module

Validates all components and their integration for proper operation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import sys

logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of a validation check."""

    def __init__(self, name: str, passed: bool, message: str, details: Optional[Dict[str, Any]] = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}

    def __repr__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status}: {self.name} - {self.message}"


class AgentValidator:
    """
    Comprehensive validation of agent components and integration.

    Performs checks on:
    - Python version
    - Required dependencies
    - API keys
    - Vector store integrity
    - Knowledge graph (if enabled)
    - Component compatibility
    """

    def __init__(self, config, debug: bool = False):
        """
        Initialize validator.

        Args:
            config: AgentConfig instance
            debug: Enable debug mode
        """
        self.config = config
        self.debug = debug
        self.results: List[ValidationResult] = []

    def validate_all(self) -> bool:
        """
        Run all validation checks.

        Returns:
            True if all checks pass, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE VALIDATION")
        logger.info("=" * 80)

        # Run all checks
        self._check_python_version()
        self._check_dependencies()
        self._check_api_keys()
        self._check_vector_store()
        self._check_knowledge_graph()
        self._check_embedding_compatibility()
        self._check_model_access()
        self._check_component_integration()

        # Summary
        passed_count = sum(1 for r in self.results if r.passed)
        failed_count = len(self.results) - passed_count

        logger.info("=" * 80)
        logger.info(f"VALIDATION SUMMARY: {passed_count}/{len(self.results)} checks passed")
        logger.info("=" * 80)

        if self.debug:
            for result in self.results:
                logger.debug(f"{result}")

        # Check for critical failures
        critical_failures = [r for r in self.results if not r.passed and "CRITICAL" in r.name]
        if critical_failures:
            logger.error(f"❌ {len(critical_failures)} CRITICAL validation failures detected")
            for failure in critical_failures:
                logger.error(f"   {failure.message}")
            return False

        # Non-critical failures are warnings
        if failed_count > 0:
            logger.warning(f"⚠️  {failed_count} non-critical validation warnings")

        logger.info("✅ Validation complete - agent ready to start")
        return True

    def _check_python_version(self):
        """Check Python version compatibility."""
        logger.debug("Checking Python version...")

        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"

        if version.major == 3 and version.minor >= 10:
            self.results.append(ValidationResult(
                name="Python Version",
                passed=True,
                message=f"Python {version_str} (compatible)",
                details={"version": version_str}
            ))
        else:
            self.results.append(ValidationResult(
                name="Python Version [CRITICAL]",
                passed=False,
                message=f"Python {version_str} - requires Python 3.10+",
                details={"version": version_str, "required": "3.10+"}
            ))

    def _check_dependencies(self):
        """Check required dependencies are installed."""
        logger.debug("Checking dependencies...")

        required_modules = [
            ("anthropic", "Claude SDK"),
            ("pydantic", "Input validation"),
            ("faiss", "Vector search - FAISS"),
            ("numpy", "Numerical operations"),
        ]

        optional_modules = [
            ("sentence_transformers", "Local embeddings (BGE-M3)"),
            ("torch", "PyTorch for local models"),
        ]

        # Check required
        for module_name, description in required_modules:
            try:
                __import__(module_name)
                self.results.append(ValidationResult(
                    name=f"Dependency: {module_name}",
                    passed=True,
                    message=f"{description} - installed",
                    details={"module": module_name}
                ))
            except ImportError:
                self.results.append(ValidationResult(
                    name=f"Dependency: {module_name} [CRITICAL]",
                    passed=False,
                    message=f"{description} - MISSING (pip install {module_name})",
                    details={"module": module_name, "required": True}
                ))

        # Check optional
        for module_name, description in optional_modules:
            try:
                __import__(module_name)
                self.results.append(ValidationResult(
                    name=f"Optional: {module_name}",
                    passed=True,
                    message=f"{description} - installed",
                    details={"module": module_name, "optional": True}
                ))
            except ImportError:
                self.results.append(ValidationResult(
                    name=f"Optional: {module_name}",
                    passed=True,  # Not critical
                    message=f"{description} - not installed (optional)",
                    details={"module": module_name, "optional": True}
                ))

    def _check_api_keys(self):
        """Check API keys are present and valid format."""
        logger.debug("Checking API keys...")

        # Anthropic API key (required)
        if self.config.anthropic_api_key:
            # Basic format validation
            if self.config.anthropic_api_key.startswith("sk-ant-"):
                self.results.append(ValidationResult(
                    name="API Key: ANTHROPIC",
                    passed=True,
                    message="Anthropic API key present (format valid)",
                    details={"key_prefix": self.config.anthropic_api_key[:10] + "..."}
                ))
            else:
                self.results.append(ValidationResult(
                    name="API Key: ANTHROPIC",
                    passed=False,
                    message="Anthropic API key has invalid format (should start with sk-ant-)",
                    details={"key_prefix": self.config.anthropic_api_key[:10] + "..."}
                ))
        else:
            self.results.append(ValidationResult(
                name="API Key: ANTHROPIC [CRITICAL]",
                passed=False,
                message="Anthropic API key missing (set ANTHROPIC_API_KEY)",
                details={}
            ))

        # OpenAI API key (for embeddings)
        import os
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key:
            if openai_key.startswith("sk-"):
                self.results.append(ValidationResult(
                    name="API Key: OPENAI",
                    passed=True,
                    message="OpenAI API key present (for embeddings)",
                    details={"key_prefix": openai_key[:10] + "..."}
                ))
            else:
                self.results.append(ValidationResult(
                    name="API Key: OPENAI",
                    passed=False,
                    message="OpenAI API key has invalid format",
                    details={}
                ))
        else:
            self.results.append(ValidationResult(
                name="API Key: OPENAI",
                passed=True,  # Not critical if using local embeddings
                message="OpenAI API key not set (required for cloud embeddings)",
                details={"required_for": "text-embedding-3-large, text-embedding-ada-002"}
            ))

    def _check_vector_store(self):
        """Check vector store exists and is valid."""
        logger.debug("Checking vector store...")

        store_path = self.config.vector_store_path

        # Check directory exists
        if not store_path.exists():
            self.results.append(ValidationResult(
                name="Vector Store Path [CRITICAL]",
                passed=False,
                message=f"Vector store not found: {store_path}",
                details={"path": str(store_path), "suggestion": "Run: python run_pipeline.py data/documents/"}
            ))
            return

        if not store_path.is_dir():
            self.results.append(ValidationResult(
                name="Vector Store Path [CRITICAL]",
                passed=False,
                message=f"Vector store path is not a directory: {store_path}",
                details={"path": str(store_path)}
            ))
            return

        # Check required files
        required_files = [
            "index_layer1.faiss",
            "index_layer2.faiss",
            "index_layer3.faiss",
        ]

        missing_files = []
        for filename in required_files:
            file_path = store_path / filename
            if not file_path.exists():
                missing_files.append(filename)

        if missing_files:
            self.results.append(ValidationResult(
                name="Vector Store Files [CRITICAL]",
                passed=False,
                message=f"Missing vector store files: {missing_files}",
                details={"missing": missing_files, "path": str(store_path)}
            ))
            return

        # Try loading vector store
        try:
            from src.hybrid_search import HybridVectorStore

            logger.debug(f"Loading vector store from {store_path}...")
            store = HybridVectorStore.load(store_path)
            stats = store.get_stats()

            # Check minimum vectors
            total_vectors = stats.get("total_vectors", 0)
            if total_vectors < 10:
                self.results.append(ValidationResult(
                    name="Vector Store Content",
                    passed=False,
                    message=f"Vector store has very few vectors ({total_vectors}) - consider indexing more documents",
                    details=stats
                ))
            else:
                self.results.append(ValidationResult(
                    name="Vector Store",
                    passed=True,
                    message=f"Vector store loaded successfully ({total_vectors} vectors)",
                    details=stats
                ))

        except Exception as e:
            self.results.append(ValidationResult(
                name="Vector Store Loading [CRITICAL]",
                passed=False,
                message=f"Failed to load vector store: {str(e)}",
                details={"path": str(store_path), "error": str(e)}
            ))

    def _check_knowledge_graph(self):
        """Check knowledge graph if enabled."""
        if not self.config.enable_knowledge_graph:
            logger.debug("Knowledge graph disabled - skipping check")
            self.results.append(ValidationResult(
                name="Knowledge Graph",
                passed=True,
                message="Knowledge graph disabled (optional)",
                details={"enabled": False}
            ))
            return

        logger.debug("Checking knowledge graph...")

        if not self.config.knowledge_graph_path:
            self.results.append(ValidationResult(
                name="Knowledge Graph Path [CRITICAL]",
                passed=False,
                message="Knowledge graph enabled but path not specified",
                details={"enabled": True, "path": None}
            ))
            return

        kg_path = self.config.knowledge_graph_path

        if not kg_path.exists():
            self.results.append(ValidationResult(
                name="Knowledge Graph File [CRITICAL]",
                passed=False,
                message=f"Knowledge graph file not found: {kg_path}",
                details={"path": str(kg_path), "suggestion": "Run: python run_pipeline.py --enable-kg"}
            ))
            return

        # Try loading KG
        try:
            from src.graph.models import KnowledgeGraph

            logger.debug(f"Loading knowledge graph from {kg_path}...")
            kg = KnowledgeGraph.load_json(str(kg_path))

            entity_count = len(kg.entities)
            rel_count = len(kg.relationships)

            if entity_count == 0:
                self.results.append(ValidationResult(
                    name="Knowledge Graph Content",
                    passed=False,
                    message="Knowledge graph is empty (no entities)",
                    details={"entities": 0, "relationships": 0}
                ))
            else:
                self.results.append(ValidationResult(
                    name="Knowledge Graph",
                    passed=True,
                    message=f"Knowledge graph loaded ({entity_count} entities, {rel_count} relationships)",
                    details={"entities": entity_count, "relationships": rel_count}
                ))

        except Exception as e:
            self.results.append(ValidationResult(
                name="Knowledge Graph Loading [CRITICAL]",
                passed=False,
                message=f"Failed to load knowledge graph: {str(e)}",
                details={"path": str(kg_path), "error": str(e)}
            ))

    def _check_embedding_compatibility(self):
        """Check embedding generator compatibility."""
        logger.debug("Checking embedding compatibility...")

        try:
            from src.embedding_generator import EmbeddingGenerator, EmbeddingConfig

            # Try creating embedder
            config = EmbeddingConfig(model="text-embedding-3-large")
            embedder = EmbeddingGenerator(config)

            self.results.append(ValidationResult(
                name="Embedding Generator",
                passed=True,
                message=f"Embedding generator initialized (model: {config.model})",
                details={"model": config.model, "dimensions": embedder.dimensions}
            ))

        except Exception as e:
            self.results.append(ValidationResult(
                name="Embedding Generator",
                passed=False,
                message=f"Failed to initialize embedding generator: {str(e)}",
                details={"error": str(e)}
            ))

    def _check_model_access(self):
        """Check Claude model accessibility."""
        logger.debug("Checking Claude model access...")

        if not self.config.anthropic_api_key:
            # Already reported in API key check
            return

        try:
            import anthropic

            # Try creating client (doesn't make API call)
            client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)

            self.results.append(ValidationResult(
                name="Claude Client",
                passed=True,
                message=f"Claude client initialized (model: {self.config.model})",
                details={"model": self.config.model}
            ))

        except Exception as e:
            self.results.append(ValidationResult(
                name="Claude Client [CRITICAL]",
                passed=False,
                message=f"Failed to initialize Claude client: {str(e)}",
                details={"error": str(e)}
            ))

    def _check_component_integration(self):
        """Check all components work together."""
        logger.debug("Checking component integration...")

        # Check tool registry
        try:
            from src.agent.tools.registry import get_registry

            registry = get_registry()
            tool_count = len(registry._tool_classes)

            if tool_count < 17:
                self.results.append(ValidationResult(
                    name="Tool Registry",
                    passed=False,
                    message=f"Expected 17 tools, found {tool_count}",
                    details={"registered_tools": tool_count, "expected": 17}
                ))
            else:
                self.results.append(ValidationResult(
                    name="Tool Registry",
                    passed=True,
                    message=f"All {tool_count} tools registered",
                    details={"registered_tools": tool_count}
                ))

        except Exception as e:
            self.results.append(ValidationResult(
                name="Tool Registry [CRITICAL]",
                passed=False,
                message=f"Failed to initialize tool registry: {str(e)}",
                details={"error": str(e)}
            ))

    def print_summary(self):
        """Print validation summary to console."""
        print("\n" + "=" * 80)
        print("🔍 AGENT VALIDATION REPORT")
        print("=" * 80)

        # Group by status
        passed = [r for r in self.results if r.passed]
        failed = [r for r in self.results if not r.passed]

        print(f"\n✅ Passed: {len(passed)}/{len(self.results)}")
        print(f"❌ Failed: {len(failed)}/{len(self.results)}")

        if failed:
            print("\n❌ FAILURES:")
            for result in failed:
                print(f"   • {result.name}: {result.message}")

        if self.debug and passed:
            print("\n✅ PASSES:")
            for result in passed:
                print(f"   • {result.name}: {result.message}")

        print("\n" + "=" * 80)
