# Evaluation Results

This document presents the evaluation results for the Enron Archive RAG system, covering retrieval performance and answer generation.

The experiments evaluate the effectiveness of different retrieval strategies, cross-encoder reranking and LoRA fine-tuning.

Results are reported from the original project evaluations. Individual evaluation records are not included in the public repository because they contain email-derived content.

## 1. Retrieval Evaluation

Retrieval was evaluated using 250 labelled questions against a corpus of 249,782 Enron emails.

Each question was associated with a labelled source email containing the expected information.

Four retrieval configurations were compared:

1. TF-IDF: Keyword-based retrieval.
2. Dense E5: Semantic retrieval using E5 embeddings.
3. Hybrid RRF: Combination of TF-IDF and E5 using Reciprocal Rank Fusion.
4. Hybrid RRF + BGE: Hybrid retrieval followed by cross-encoder reranking.

### Retrieval Performance

| Method | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|
| TF-IDF | 20.0% | 42.8% | 52.8% | 0.306 |
| Dense E5 | 26.4% | 55.2% | 62.8% | 0.389 |
| Hybrid RRF | 37.2% | 67.6% | 73.6% | 0.502 |
| Hybrid RRF + BGE | **47.6%** | **73.6%** | **78.8%** | **0.588** |

### Evaluation Metrics

**Recall@K**

Measures the proportion of questions for which the labelled source email appears within the top K retrieved results.

For example, Recall@10 of 78.8% means that the labelled source email appeared among the first 10 results for 197 of the 250 evaluation questions.

**Mean Reciprocal Rank (MRR)**

Measures how highly the labelled source email is ranked.

For each question, the reciprocal rank is calculated as:

1 / Rank of the labelled source email

The final MRR is the average across all evaluation questions, with a score of zero when the source is not retrieved.

Higher values indicate that relevant source emails are generally ranked closer to the top.

### Key Findings

Dense E5 retrieval outperformed TF-IDF across all measured metrics, demonstrating the benefits of semantic representations for retrieving relevant emails.

Combining the two approaches using Reciprocal Rank Fusion improved Recall@10 from 62.8% to 73.6%, indicating that lexical and semantic retrieval provide complementary results.

Adding BGE cross-encoder reranking further improved performance:

- Recall@1 increased from 37.2% to 47.6%.
- Recall@5 increased from 67.6% to 73.6%.
- Recall@10 increased from 73.6% to 78.8%.
- MRR increased from 0.502 to 0.588.

The improvement in Recall@1 is particularly useful because it indicates that reranking placed the labelled source email first more frequently.

Overall, the results demonstrate the value of combining sparse retrieval, dense embeddings and cross-encoder reranking rather than relying on a single retrieval method.

## 2. Generation Evaluation

Generation was evaluated using 50 questions for each model configuration.

Two configurations were compared:

- Base Qwen3-4B-Instruct-2507.
- Qwen3-4B with experimental LoRA fine-tuning.

Both configurations were evaluated within the RAG pipeline.

### Generation Performance

| Metric | Base Qwen | LoRA Qwen |
|---|---:|---:|
| Answer Accuracy | **76.0% (38/50)** | 46.0% (23/50) |
| Citation Validity | **92.0% (46/50)** | 82.0% (41/50) |
| Faithfulness | **82.6% (38/46)** | 73.2% (30/41) |

### Evaluation Metrics

**Answer Accuracy**

Measures whether the generated answer includes the information required to answer the question.

Responses were assessed using binary AI-assigned scores.

**Citation Validity**

Measures whether a generated citation refers to a source contained within the retrieved evidence.

A valid citation does not necessarily mean that the cited source matches the labelled ground-truth email or fully supports the answer.

**Faithfulness**

Measures whether the generated answer is supported by the cited evidence.

Only responses containing valid citations were included in the faithfulness calculation. Consequently, faithfulness uses a different denominator from answer accuracy.

### Key Findings

The base Qwen model outperformed the LoRA configuration across all three evaluation metrics.

Answer accuracy decreased from 76.0% to 46.0% following fine-tuning.

One possible explanation is the difference between the fine-tuning and inference tasks.

The LoRA model was trained using synthetic question-answer examples derived from individual emails, whereas the RAG pipeline generates answers using multiple retrieved evidence chunks.

This difference may have limited the model's ability to generalise from the fine-tuning examples to the final question-answering task.

These results do not establish that LoRA is unsuitable for RAG systems generally. They indicate that the particular training configuration used in this experiment did not improve performance.

The final Streamlit application therefore uses the base Qwen model.

## 3. Evaluation Limitations

The evaluation has several limitations.

**Retrieval evaluation**

Each question has one labelled source email. Other emails may also contain relevant information, meaning the evaluation does not capture every potentially valid retrieval result.

**Generation evaluation**

The generation evaluation contains 50 questions per configuration, limiting the conclusions that can be drawn about performance across the entire archive.

**AI-based scoring**

Answer accuracy and faithfulness were assessed using externally assigned AI-judge scores rather than independent human annotations.

The original evaluation workflow does not preserve the complete judge prompts or judge model configuration.

**Generalisability**

The reported results apply to this dataset, retrieval configuration and evaluation questions. They should not be interpreted as general performance guarantees.

## 4. Reproducing the Evaluation

The evaluation methodology is documented in the research notebooks:

- `notebooks/03_retrieval/Hybrid-Retrieval-Final.ipynb`
- `notebooks/04_generation/Generation-Evaluation.ipynb`
- `notebooks/05_fine_tuning/LoRa-Evaluation.ipynb`

These paths are relative to the repository root.

The notebooks contain the relevant evaluation code and experimental configurations.

Reproducing the full evaluation requires the Enron corpus, precomputed indexes, evaluation questions and model dependencies.

The individual evaluation CSV files are retained locally and excluded from the public repository because they contain email-derived information.

See the [Research Notebook Guide](../notebooks/README.md) for further information.