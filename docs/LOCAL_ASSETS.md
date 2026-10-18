# Local Assets

The Enron Archive RAG application requires several datasets and
precomputed retrieval indexes that are not included in the public
GitHub repository.

These files are excluded because of their size and, where applicable,
data redistribution considerations.

The application expects the following assets to be available
relative to the repository root.

## Required Application Assets

| Path | Purpose |
|---|---|
| `enron_clean.csv` | Cleaned Enron email corpus used for retrieving original source emails |
| `full_corpus_tfidf_matrix.npz` | Precomputed sparse TF-IDF matrix |
| `full_corpus_tfidf_vectorizer.joblib` | Fitted TF-IDF vectorizer used to process queries |
| `e5_embeddings_final/embedding_manifest.json` | Metadata describing the embedding index |
| `e5_embeddings_final/e5_dense_index.faiss` | FAISS index used for dense semantic retrieval |
| `e5_embeddings_final/shards/*_chunks.parquet` | Chunk metadata required to associate retrieved evidence with original emails |

The final dataset contains 249,782 emails and 599,446 chunks,
distributed across 250 chunk metadata shards.

The corpus, chunk metadata and retrieval indexes must correspond
to the same dataset and preprocessing configuration.

Mismatched identifiers, row ordering or index configurations may
produce incorrect retrieval results or source references.

## Asset Verification

The repository includes a script for checking whether the required
local assets are present.

From the repository root, run:

```bash
python scripts/check_assets.py
```

The script checks for required assets before running the application.

This check does not guarantee that the indexes and dataset are
mutually compatible or that the complete application will run.

## Rebuilding the Retrieval Indexes

The repository includes research notebooks documenting the
preparation of the Enron corpus and construction of the retrieval
indexes.

Relevant notebooks include:

- [Dataset Loader](../notebooks/01_data_preparation/Dataset-Loader.ipynb)
- [Final Embeddings](../notebooks/02_chunking_embeddings/Embeddings-Final.ipynb)
- [Final Hybrid Retrieval](../notebooks/03_retrieval/Hybrid-Retrieval-Final.ipynb)

Rebuilding the indexes requires obtaining the original Enron dataset,
processing the email corpus and generating the corresponding
TF-IDF and E5 representations.

The complete rebuilding workflow has not been independently
verified from a fresh installation.

### Embedding Shards

The embedding manifest references intermediate `.npy` files
containing dense embedding vectors.

These files are not required for inference when the corresponding
precomputed FAISS index is available.

However, regenerating or rebuilding the dense index may require
recreating these embeddings.

## Experimental LoRA Adapter

The project also includes an experimental LoRA fine-tuning workflow.

The resulting adapter is stored locally in:

```text
qwen_enron_lora/
```

This directory is excluded from the public GitHub repository.

The final Streamlit application uses the base Qwen model and
does not require the LoRA adapter.

The fine-tuning experiment and its evaluation are documented in:

[LoRA Evaluation](../notebooks/05_fine_tuning/LoRa-Evaluation.ipynb)

## Data and Security Considerations

The Enron dataset contains historical enterprise communications
that may include personal information.

Anyone recreating or distributing the corpus and derived indexes
should consider the dataset's source terms and the implications
of redistributing email-derived information.

Serialized Python objects, including joblib files, should only
be loaded from trusted sources.

## Additional Documentation

- [Main README](../README.md)
- [Research Notebooks](../notebooks/README.md)
- [Evaluation Results](../results/README.md)