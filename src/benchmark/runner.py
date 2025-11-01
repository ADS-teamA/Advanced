"""
Benchmark runner - main orchestrator.

Coordinates:
1. Dataset loading
2. Agent initialization
3. Query evaluation loop
4. Metric computation
5. Cost tracking
6. Progress reporting
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from tqdm import tqdm

from ..agent.agent_core import AgentCore
from ..agent.config import AgentConfig
from ..agent.prompt_loader import load_prompt
from ..hybrid_search import HybridVectorStore
from ..cost_tracker import get_global_tracker, reset_global_tracker

from .config import BenchmarkConfig
from .dataset import BenchmarkDataset, QueryExample
from .metrics import compute_all_metrics, aggregate_metrics, format_metrics

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """
    Result for a single query evaluation.

    Attributes:
        query_id: Unique query identifier
        query: Question text
        predicted_answer: Model's answer
        ground_truth_answers: List of expected answers
        metrics: Dict of metric scores
        retrieval_time_ms: Time to retrieve + generate answer
        cost_usd: API cost for this query
        rag_confidence: RAG confidence score (0-1) or None if unavailable
    """

    query_id: int
    query: str
    predicted_answer: str
    ground_truth_answers: List[str]
    metrics: Dict[str, float]
    retrieval_time_ms: float
    cost_usd: float
    rag_confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON export."""
        result = {
            "query_id": self.query_id,
            "query": self.query,
            "predicted_answer": self.predicted_answer,
            "ground_truth_answers": self.ground_truth_answers,
            "metrics": self.metrics,
            "retrieval_time_ms": round(self.retrieval_time_ms, 2),
            "cost_usd": round(self.cost_usd, 6),
        }

        # Add RAG confidence if available
        if self.rag_confidence is not None:
            result["rag_confidence"] = round(self.rag_confidence, 4)

        return result


@dataclass
class BenchmarkResult:
    """
    Complete benchmark evaluation results.

    Attributes:
        dataset_name: Name of dataset evaluated
        total_queries: Number of queries evaluated
        aggregate_metrics: Mean scores per metric
        query_results: Per-query detailed results
        total_time_seconds: Total evaluation time
        total_cost_usd: Total API cost
        config: Benchmark configuration
        timestamp: When evaluation was run
    """

    dataset_name: str
    total_queries: int
    aggregate_metrics: Dict[str, float]
    query_results: List[QueryResult]
    total_time_seconds: float
    total_cost_usd: float
    config: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON export."""
        return {
            "dataset_name": self.dataset_name,
            "total_queries": self.total_queries,
            "aggregate_metrics": self.aggregate_metrics,
            "query_results": [qr.to_dict() for qr in self.query_results],
            "total_time_seconds": round(self.total_time_seconds, 2),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_time_per_query_ms": round(
                (self.total_time_seconds * 1000) / self.total_queries, 2
            )
            if self.total_queries > 0
            else 0,
            "cost_per_query_usd": round(self.total_cost_usd / self.total_queries, 6)
            if self.total_queries > 0
            else 0,
            "config": self.config,
            "timestamp": self.timestamp,
        }


class BenchmarkRunner:
    """
    Main benchmark orchestrator.

    Usage:
        config = BenchmarkConfig.from_env()
        runner = BenchmarkRunner(config)
        result = runner.run()
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config

        # Load dataset
        logger.info(f"Loading dataset from {config.dataset_path}")
        self.dataset = BenchmarkDataset.from_json(config.dataset_path)

        # Print dataset statistics
        stats = self.dataset.get_statistics()
        logger.info(f"Dataset statistics: {stats['total_queries']} queries, "
                   f"{stats['unique_source_documents']} source documents")

        # Load vector store
        logger.info(f"Loading vector store from {config.vector_store_path}")
        self.vector_store = HybridVectorStore.load(config.vector_store_path)

        # Initialize agent
        logger.info("Initializing RAG agent")
        self.agent = self._initialize_agent()

        logger.info("Benchmark runner initialized successfully")

    def _initialize_agent(self) -> AgentCore:
        """
        Initialize agent with benchmark-specific configuration.

        Returns:
            AgentCore instance configured for benchmark evaluation
        """
        # Create agent config with benchmark settings
        agent_config = AgentConfig.from_env(
            model=self.config.agent_model,
            temperature=self.config.agent_temperature,
            enable_prompt_caching=self.config.enable_prompt_caching,
            vector_store_path=Path(self.config.vector_store_path),
            debug_mode=self.config.debug_mode,
        )

        # Override tool config for retrieval parameters
        agent_config.tool_config.default_k = self.config.k
        agent_config.tool_config.enable_reranking = self.config.enable_reranking
        agent_config.tool_config.enable_graph_boost = self.config.enable_graph_boost

        # Initialize pipeline components (required for tool registry)
        from ..embedding_generator import EmbeddingGenerator, EmbeddingConfig
        from ..reranker import CrossEncoderReranker
        from ..context_assembly import ContextAssembler, CitationFormat
        from ..agent.tools.registry import get_registry

        # Create embedder
        logger.info(f"Initializing embedder: {agent_config.embedding_model}")
        embedder = EmbeddingGenerator(
            EmbeddingConfig(
                model=agent_config.embedding_model, batch_size=100, normalize=True
            )
        )

        # Create reranker (if enabled)
        reranker = None
        if self.config.enable_reranking:
            logger.info(f"Initializing reranker: {agent_config.tool_config.reranker_model}")
            try:
                reranker = CrossEncoderReranker(
                    model_name=agent_config.tool_config.reranker_model
                )
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}. Continuing without reranking.")
                agent_config.tool_config.enable_reranking = False

        # Create context assembler
        context_assembler = ContextAssembler(citation_format=CitationFormat.INLINE)

        # Initialize tool registry BEFORE creating AgentCore
        logger.info("Initializing tool registry...")
        registry = get_registry()
        registry.initialize_tools(
            vector_store=self.vector_store,
            embedder=embedder,
            reranker=reranker,
            graph_retriever=None,  # No graph for benchmark
            knowledge_graph=None,  # No graph for benchmark
            context_assembler=context_assembler,
            config=agent_config.tool_config,
        )

        logger.info(f"Tool registry initialized: {len(registry)} tools available")

        # Create agent core (registry already initialized with tools)
        agent = AgentCore(agent_config)

        # Load and append benchmark prompt to agent prompt (preserve tool selection strategy)
        benchmark_prompt = load_prompt("agent_benchmark_prompt")
        agent.config.system_prompt = (
            f"{agent.config.system_prompt}\n\n"
            f"---\n\n"
            f"{benchmark_prompt}"
        )
        logger.info("Benchmark system prompt appended to agent prompt (loaded from prompts/agent_benchmark_prompt.txt)")

        # Initialize with documents (loads document list into context)
        agent.initialize_with_documents()

        logger.info(
            f"Agent initialized: {agent_config.model}, "
            f"k={self.config.k}, "
            f"reranking={'enabled' if self.config.enable_reranking else 'disabled'}"
        )

        return agent

    def _evaluate_query(self, query_example: QueryExample) -> QueryResult:
        """
        Evaluate single query.

        Args:
            query_example: Query to evaluate

        Returns:
            QueryResult with metrics and timing

        Raises:
            RuntimeError: If agent fails (when fail_fast=True)
        """
        query_text = query_example.query
        expected_answers = query_example.get_expected_answers()

        # Reset conversation for independent query
        self.agent.reset_conversation()

        # Track cost before query
        tracker = get_global_tracker()
        cost_before = tracker.get_total_cost()

        # Measure time
        start_time = time.time()

        try:
            # Get agent response (non-streaming for consistency)
            response_text = self.agent.process_message(query_text, stream=False)

        except Exception as e:
            error_msg = f"Agent failed for query {query_example.query_id}: {e}"

            if self.config.fail_fast:
                # Stop entire benchmark
                raise RuntimeError(error_msg) from e
            else:
                # Log and continue with empty answer
                logger.error(error_msg)
                response_text = ""

        # Calculate elapsed time
        elapsed_ms = (time.time() - start_time) * 1000

        # Calculate cost for this query
        cost_after = tracker.get_total_cost()
        query_cost = cost_after - cost_before

        # Extract answer from <ANSWER></ANSWER> tags
        import re
        answer_match = re.search(r'<ANSWER>(.*?)</ANSWER>', response_text, re.DOTALL)
        if answer_match:
            extracted_answer = answer_match.group(1).strip()
            logger.debug(f"Extracted answer from tags: {extracted_answer[:100]}...")
        else:
            # Fallback: use full response if tags not found
            extracted_answer = response_text
            logger.warning(f"No <ANSWER> tags found for query {query_example.query_id}, using full response")

        # Compute metrics using extracted answer
        metrics = compute_all_metrics(extracted_answer, expected_answers)

        # Extract RAG confidence from last tool call (if available)
        rag_confidence = None
        try:
            confidence_data = self.agent.get_latest_rag_confidence()
            if confidence_data:
                rag_confidence = confidence_data.get("overall_confidence")
                if self.config.debug_mode:
                    logger.debug(
                        f"RAG Confidence: {rag_confidence:.3f} "
                        f"({confidence_data.get('interpretation', 'Unknown')})"
                    )
        except Exception as e:
            logger.warning(f"Failed to extract RAG confidence: {e}")

        return QueryResult(
            query_id=query_example.query_id,
            query=query_text,
            predicted_answer=extracted_answer,  # Use extracted answer, not full response
            ground_truth_answers=expected_answers,
            metrics=metrics,
            retrieval_time_ms=elapsed_ms,
            cost_usd=query_cost,
            rag_confidence=rag_confidence,
        )

    def run(self) -> BenchmarkResult:
        """
        Run complete benchmark evaluation.

        Returns:
            BenchmarkResult with aggregated metrics and per-query results

        Raises:
            RuntimeError: If evaluation fails (when fail_fast=True)
        """
        logger.info("=" * 80)
        logger.info("STARTING BENCHMARK EVALUATION")
        logger.info("=" * 80)

        # Reset cost tracker for this benchmark run
        reset_global_tracker()
        tracker = get_global_tracker()

        # Get queries to evaluate
        queries = self.dataset.get_queries(max_queries=self.config.max_queries)
        logger.info(f"Evaluating {len(queries)} queries")

        # Evaluation loop with progress bar
        start_time = time.time()
        query_results = []

        progress_bar = tqdm(
            queries,
            desc="Evaluating queries",
            unit="query",
            disable=self.config.debug_mode,  # Disable in debug mode
        )

        for query_example in progress_bar:
            try:
                result = self._evaluate_query(query_example)
                query_results.append(result)

                # Update progress bar with metrics
                progress_bar.set_postfix({
                    "EM": f"{result.metrics.get('exact_match', 0):.2f}",
                    "F1": f"{result.metrics.get('f1_score', 0):.2f}",
                })

                # Log per-query results in debug mode
                if self.config.debug_mode:
                    logger.info(
                        f"[{result.query_id}/{len(queries)}] "
                        f"{format_metrics(result.metrics)} | "
                        f"Time: {result.retrieval_time_ms:.0f}ms | "
                        f"Cost: ${result.cost_usd:.6f}"
                    )

                # Rate limiting delay (for API quotas like Gemini Free Tier)
                if self.config.rate_limit_delay > 0:
                    time.sleep(self.config.rate_limit_delay)

            except Exception as e:
                # This should only happen if fail_fast=False
                logger.error(f"Skipping query {query_example.query_id}: {e}")
                continue

        # Calculate total time and cost
        total_time = time.time() - start_time
        total_cost = tracker.get_total_cost()

        # Aggregate metrics
        all_metrics = [qr.metrics for qr in query_results]
        aggregated = aggregate_metrics(all_metrics)

        # Aggregate RAG confidence (if available)
        confidence_scores = [
            qr.rag_confidence for qr in query_results if qr.rag_confidence is not None
        ]
        if confidence_scores:
            aggregated["rag_confidence"] = sum(confidence_scores) / len(confidence_scores)
            logger.info(
                f"Average RAG Confidence: {aggregated['rag_confidence']:.4f} "
                f"({len(confidence_scores)}/{len(query_results)} queries)"
            )

        # Create result object
        result = BenchmarkResult(
            dataset_name="PrivacyQA",
            total_queries=len(query_results),
            aggregate_metrics=aggregated,
            query_results=query_results,
            total_time_seconds=total_time,
            total_cost_usd=total_cost,
            config=self.config.to_dict(),
        )

        logger.info("=" * 80)
        logger.info("BENCHMARK COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Queries evaluated: {result.total_queries}")
        logger.info(f"Total time: {result.total_time_seconds:.1f}s")
        logger.info(f"Total cost: ${result.total_cost_usd:.4f}")
        logger.info(f"Aggregate metrics: {format_metrics(aggregated, precision=3)}")

        return result
