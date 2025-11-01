# Benchmark Report: PrivacyQA

**Timestamp:** 2025-11-01T10:00:04.814076
**Total Queries:** 3
**Total Time:** 28.0s
**Total Cost:** $0.0195

## Aggregate Metrics

| Metric | Score |
|--------|-------|
| Combined F1 | 0.3420 |
| Embedding Similarity | 0.5634 |
| Exact Match (EM) | 0.3333 |
| F1 Score | 0.4577 |
| Precision | 0.5812 |
| RAG Confidence (Avg) | 0.3900 |
| Recall | 0.4301 |

## Performance Metrics

| Metric | Value |
|--------|-------|
| Avg Time per Query | 9346ms |
| Cost per Query | $0.006506 |

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
  "max_queries": 3,
  "debug_mode": false,
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
  "save_per_query": false
}
```

## Per-Query Results

### Top 3 Best Results (by F1 Score)

| Query ID | F1 | EM | Precision | Recall | RAG Conf | Time (ms) | Query Preview |
|----------|----|----|-----------|--------|----------|-----------|---------------|
| 3 | 1.000 | 1 | 1.000 | 1.000 | N/A | 5578 | Consider "Fiverr"'s privacy policy; what type of identifiabl... |
| 1 | 0.282 | 0 | 0.344 | 0.239 | N/A | 8519 | Consider "Fiverr"'s privacy policy; who can see which tasks ... |
| 2 | 0.091 | 0 | 0.400 | 0.051 | 0.390 | 11251 | Consider "Fiverr"'s privacy policy; who can see the jobs tha... |

### Bottom 3 Worst Results (by F1 Score)

| Query ID | F1 | EM | Precision | Recall | RAG Conf | Time (ms) | Query Preview |
|----------|----|----|-----------|--------|----------|-----------|---------------|
| 3 | 1.000 | 1 | 1.000 | 1.000 | N/A | 5578 | Consider "Fiverr"'s privacy policy; what type of identifiabl... |
| 1 | 0.282 | 0 | 0.344 | 0.239 | N/A | 8519 | Consider "Fiverr"'s privacy policy; who can see which tasks ... |
| 2 | 0.091 | 0 | 0.400 | 0.051 | 0.390 | 11251 | Consider "Fiverr"'s privacy policy; who can see the jobs tha... |

## Metric Interpretation

- **Exact Match (EM)**: 1.0 if prediction exactly matches any ground truth, else 0.0
- **F1 Score**: Harmonic mean of precision and recall (token-level)
- **Precision**: Fraction of predicted tokens that are correct
- **Recall**: Fraction of ground truth tokens that were predicted
- **RAG Confidence**: Retrieval quality score (0-1). ≥0.85: High (automated), 0.70-0.84: Medium (review recommended), 0.50-0.69: Low (mandatory review), <0.50: Very low (expert required)

---
*Generated on 2025-11-01 10:00:04*