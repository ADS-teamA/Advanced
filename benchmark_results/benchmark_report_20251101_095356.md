# Benchmark Report: PrivacyQA

**Timestamp:** 2025-11-01T09:53:56.545166
**Total Queries:** 1
**Total Time:** 1.7s
**Total Cost:** $0.0020

## Aggregate Metrics

| Metric | Score |
|--------|-------|
| Combined F1 | 0.0000 |
| Embedding Similarity | 0.0000 |
| Exact Match (EM) | 0.0000 |
| F1 Score | 0.0000 |
| Precision | 0.0000 |
| Recall | 0.0000 |

## Performance Metrics

| Metric | Value |
|--------|-------|
| Avg Time per Query | 1689ms |
| Cost per Query | $0.002000 |

## Configuration

```json
{
  "dataset_path": "benchmark_dataset/privacy_qa.json",
  "documents_dir": "benchmark_dataset/privacy_qa",
  "vector_store_path": "benchmark_db/combined_store",
  "k": 5,
  "enable_reranking": true,
  "enable_graph_boost": false,
  "enable_hybrid_search": true,
  "max_queries": 1,
  "debug_mode": true,
  "fail_fast": true,
  "agent_model": "gemini-2.5-flash",
  "agent_temperature": 0.0,
  "enable_prompt_caching": true,
  "metrics": [
    "exact_match",
    "f1_score",
    "precision",
    "recall"
  ],
  "output_dir": "benchmark_results",
  "save_markdown": true,
  "save_json": true,
  "save_per_query": true
}
```

## Per-Query Results

### Top 1 Best Results (by F1 Score)

| Query ID | F1 | EM | Precision | Recall | RAG Conf | Time (ms) | Query Preview |
|----------|----|----|-----------|--------|----------|-----------|---------------|
| 1 | 0.000 | 0 | 0.000 | 0.000 | N/A | 1660 | Consider "Fiverr"'s privacy policy; who can see which tasks ... |

### Bottom 1 Worst Results (by F1 Score)

| Query ID | F1 | EM | Precision | Recall | RAG Conf | Time (ms) | Query Preview |
|----------|----|----|-----------|--------|----------|-----------|---------------|
| 1 | 0.000 | 0 | 0.000 | 0.000 | N/A | 1660 | Consider "Fiverr"'s privacy policy; who can see which tasks ... |

## Metric Interpretation

- **Exact Match (EM)**: 1.0 if prediction exactly matches any ground truth, else 0.0
- **F1 Score**: Harmonic mean of precision and recall (token-level)
- **Precision**: Fraction of predicted tokens that are correct
- **Recall**: Fraction of ground truth tokens that were predicted
- **RAG Confidence**: Retrieval quality score (0-1). ≥0.85: High (automated), 0.70-0.84: Medium (review recommended), 0.50-0.69: Low (mandatory review), <0.50: Very low (expert required)

---
*Generated on 2025-11-01 09:53:56*