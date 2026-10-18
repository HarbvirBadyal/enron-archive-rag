# Research Notebooks

This directory contains the research and experiments conducted during
the development of the Enron Archive RAG system.

The notebooks document the progression from dataset preparation and
embedding model selection to hybrid retrieval, answer generation
and LoRA fine-tuning.

The final application is implemented separately in the repository's
Python files. Running every notebook is not necessary to use the
Streamlit application.

## Notebook Overview

| Stage | Notebook | Purpose |
|---|---|---|
| Data Preparation | [Dataset Loader](01_data_preparation/Dataset-Loader.ipynb) | Load, clean and prepare the Enron email archive |
| Chunking & Embeddings | [Chunking Evaluation](02_chunking_embeddings/Chunking-Evaluation.ipynb) | Evaluate chunk sizes and overlap configurations |
| Chunking & Embeddings | [Embedding Evaluation](02_chunking_embeddings/Embedding-Evaluation.ipynb) | Compare embedding models for semantic retrieval |
| Chunking & Embeddings | [Final Embeddings](02_chunking_embeddings/Embeddings-Final.ipynb) | Generate full-corpus E5 embeddings and FAISS index |
| Chunking & Embeddings | [E5 Evaluation](02_chunking_embeddings/E5-Evaluation.ipynb) | Evaluate dense retrieval performance |
| Retrieval | [TF-IDF Evaluation](03_retrieval/TF-IDF-Evaluation.ipynb) | Build and evaluate the lexical retrieval baseline |
| Retrieval | [Hybrid Evaluation](03_retrieval/Hybrid-Evaluation.ipynb) | Experiment with retrieval fusion and reranking |
| Retrieval | [Final Hybrid Evaluation](03_retrieval/Hybrid-Retrieval-Final.ipynb) | Evaluate the final hybrid retrieval configuration |
| Retrieval | [Final Retrieval](03_retrieval/Retrieval-Final.ipynb) | Implement the reusable retrieval pipeline |
| Generation | [Final Generation](04_generation/Generation-Final.ipynb) | Implement evidence-grounded generation using Qwen |
| Generation | [Generation Evaluation](04_generation/Generation-Evaluation.ipynb) | Evaluate answer accuracy, citations and faithfulness |
| Fine-Tuning | [LoRA Evaluation](05_fine_tuning/LoRa-Evaluation.ipynb) | Train and evaluate an experimental LoRA adapter |

## Experimental Workflow

The project was developed through five main stages.

### 1. Data Preparation

The Enron email archive was loaded, cleaned and prepared for
subsequent retrieval experiments.

### 2. Chunking and Embedding Selection

Different chunk sizes, overlap configurations and embedding models
were evaluated to identify an effective semantic representation.

The final configuration uses E5-base-v2 embeddings with
320-token chunks and 64-token overlap.

### 3. Retrieval and Reranking

The retrieval experiments compare TF-IDF, dense embeddings
and hybrid retrieval using Reciprocal Rank Fusion.

BGE cross-encoder reranking was subsequently evaluated
to improve the ranking of retrieved evidence.

### 4. Answer Generation

The generation notebooks implement locally hosted Qwen3-4B
inference using retrieved email evidence.

The system generates answers with source references,
allowing the original emails to be inspected.

### 5. LoRA Fine-Tuning

An experimental LoRA adapter was trained using synthetic
question-answer examples.

The fine-tuned configuration was evaluated against the
base Qwen model.

The base model achieved better results in the project evaluation
and is used in the final application.

## Running the Notebooks

Install the research dependencies from the repository root:

```bash
python -m pip install -r requirements-research.txt
```

Launch Jupyter:

```bash
python -m jupyterlab
```

Run the first setup cell in each notebook before executing
the remaining cells.

The setup cells resolve the repository root so that datasets,
indexes and other assets are accessed using consistent paths.

## Rebuilding the Pipeline

The intended main workflow is:

1. `Dataset-Loader.ipynb`
2. `Embeddings-Final.ipynb`
3. `Hybrid-Retrieval-Final.ipynb`
4. `Retrieval-Final.ipynb`
5. `Generation-Final.ipynb`

The other notebooks document supporting experiments and comparisons.

This sequence describes the development workflow rather than
a fully verified, one-command reproduction process.

Rebuilding the full pipeline requires the original Enron dataset,
evaluation files, model dependencies and sufficient computing resources.

Some intermediate datasets and embedding files are not included
in the public repository.

See [Local Assets](../docs/LOCAL_ASSETS.md) for further information.

**Important:** Some notebooks generate or overwrite datasets and
retrieval indexes. Back up existing assets before rerunning
data preparation or index-building experiments.

## Evaluation

Retrieval and generation evaluations are documented in their
respective notebooks.

The reported generation metrics use externally assigned
AI-judge scores. Running the generation notebooks alone
does not reproduce those judgments.

See [Evaluation Results](../results/README.md)
for the reported metrics, methodology and limitations.