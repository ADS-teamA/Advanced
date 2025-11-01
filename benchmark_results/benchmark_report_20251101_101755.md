# Benchmark Report: PrivacyQA

**Timestamp:** 2025-11-01T10:17:55.109275
**Total Queries:** 194
**Total Time:** 79.5s
**Total Cost:** $0.0343

## Aggregate Metrics

| Metric | Score |
|--------|-------|
| Combined F1 | 0.1158 |
| Embedding Similarity | 0.1481 |
| Exact Match (EM) | 0.0000 |
| F1 Score | 0.1215 |
| Precision | 0.1110 |
| RAG Confidence (Avg) | 0.3595 |
| Recall | 0.1724 |

## Performance Metrics

| Metric | Value |
|--------|-------|
| Avg Time per Query | 410ms |
| Cost per Query | $0.000177 |

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
  "max_queries": null,
  "debug_mode": false,
  "fail_fast": true,
  "rate_limit_delay": 0.0,
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

### Top 10 Best Results (by F1 Score)

| Query ID | F1 | EM | Precision | Recall | RAG Conf | Time (ms) | Query Preview |
|----------|----|----|-----------|--------|----------|-----------|---------------|
| 3 | 0.821 | 0 | 0.696 | 1.000 | N/A | 8861 | Consider "Fiverr"'s privacy policy; what type of identifiabl... |
| 6 | 0.420 | 0 | 0.808 | 0.284 | 0.310 | 12209 | Consider "Fiverr"'s privacy policy; what information does th... |
| 2 | 0.310 | 0 | 0.344 | 0.282 | 0.196 | 7463 | Consider "Fiverr"'s privacy policy; who can see the jobs tha... |
| 5 | 0.277 | 0 | 0.226 | 0.359 | N/A | 9199 | Consider "Fiverr"'s privacy policy; who can see my informati... |
| 4 | 0.277 | 0 | 0.462 | 0.198 | 0.575 | 6764 | Consider "Fiverr"'s privacy policy; how is my info protected... |
| 147 | 0.161 | 0 | 0.132 | 0.208 | N/A | 58 | Consider "23andMe"'s privacy policy; is any information reco... |
| 180 | 0.157 | 0 | 0.143 | 0.175 | N/A | 62 | Consider "Viber Messenger"'s privacy policy; can anyone view... |
| 187 | 0.157 | 0 | 0.143 | 0.175 | N/A | 65 | Consider "Viber Messenger"'s privacy policy; can anyone view... |
| 143 | 0.155 | 0 | 0.169 | 0.143 | N/A | 69 | Consider "23andMe"'s privacy policy; are you accessing and i... |
| 150 | 0.152 | 0 | 0.105 | 0.276 | N/A | 55 | Consider "23andMe"'s privacy policy; will my test results be... |

### Bottom 10 Worst Results (by F1 Score)

| Query ID | F1 | EM | Precision | Recall | RAG Conf | Time (ms) | Query Preview |
|----------|----|----|-----------|--------|----------|-----------|---------------|
| 185 | 0.079 | 0 | 0.052 | 0.167 | N/A | 63 | Consider "Viber Messenger"'s privacy policy; can my call log... |
| 1 | 0.078 | 0 | 0.400 | 0.043 | 0.357 | 5442 | Consider "Fiverr"'s privacy policy; who can see which tasks ... |
| 145 | 0.078 | 0 | 0.052 | 0.160 | N/A | 66 | Consider "23andMe"'s privacy policy; will you destroy my dna... |
| 148 | 0.078 | 0 | 0.053 | 0.154 | N/A | 63 | Consider "23andMe"'s privacy policy; where is the informatio... |
| 65 | 0.076 | 0 | 0.065 | 0.091 | N/A | 67 | Consider "Groupon"'s privacy policy; what does groupon do wi... |
| 121 | 0.067 | 0 | 0.039 | 0.250 | N/A | 63 | Consider "TickTick: To Do List with Reminder, Day Planner"'s... |
| 30 | 0.065 | 0 | 0.039 | 0.200 | N/A | 83 | Consider "Keep"'s privacy policy; does the app access my con... |
| 33 | 0.065 | 0 | 0.039 | 0.200 | N/A | 77 | Consider "Keep"'s privacy policy; do you access any of my co... |
| 22 | 0.065 | 0 | 0.039 | 0.188 | N/A | 67 | Consider "Keep"'s privacy policy; will my progress only be p... |
| 108 | 0.022 | 0 | 0.013 | 0.062 | N/A | 70 | Consider "Wordscapes"'s privacy policy; are there any advert... |

## Metric Interpretation

- **Exact Match (EM)**: 1.0 if prediction exactly matches any ground truth, else 0.0
- **F1 Score**: Harmonic mean of precision and recall (token-level)
- **Precision**: Fraction of predicted tokens that are correct
- **Recall**: Fraction of ground truth tokens that were predicted
- **RAG Confidence**: Retrieval quality score (0-1). ≥0.85: High (automated), 0.70-0.84: Medium (review recommended), 0.50-0.69: Low (mandatory review), <0.50: Very low (expert required)

---
*Generated on 2025-11-01 10:17:55*