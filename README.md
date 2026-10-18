# Enron Archive RAG
### Enterprise Email Search and Question Answering

An end-to-end Retrieval-Augmented Generation (RAG) system designed to
answer natural-language questions over approximately 250,000 Enron emails.

The system combines traditional information retrieval, semantic search,
cross-encoder reranking and a locally hosted large language model to
generate evidence-grounded answers with citations to original emails.

Developed as part of my MSc Data Science and Artificial Intelligence
at Queen's University Belfast.

## Project Overview

Searching large collections of unstructured emails is challenging because
relevant information may be distributed across thousands of messages,
expressed using different terminology or buried within lengthy conversations.

This project investigates how different retrieval and generation techniques
can be combined to improve question answering over an enterprise email archive.

The system processes 249,782 emails and uses a hybrid retrieval pipeline
to identify relevant evidence before generating an answer.

Key features include:

- Hybrid retrieval combining TF-IDF and E5 dense embeddings.
- Reciprocal Rank Fusion (RRF) to combine retrieval results.
- BGE cross-encoder reranking to improve candidate relevance.
- Local answer generation using Qwen3-4B.
- Source-linked responses allowing users to inspect original emails.
- A Streamlit interface for interactive question answering.
- Experimental evaluation of retrieval, generation and LoRA fine-tuning.

The final application uses the base Qwen model. LoRA fine-tuning was
investigated separately as an experimental approach.

## System Architecture

The RAG pipeline consists of four main stages.

### 1. Data Preparation and Embeddings

The Enron corpus is cleaned and divided into overlapping text chunks.

Experiments were conducted to investigate different chunk sizes,
overlap configurations and embedding models.

The final configuration uses:

- Dataset: 249,782 Enron emails
- Chunk size: 320 tokens
- Chunk overlap: 64 tokens
- Dense embedding model: intfloat/e5-base-v2
- Vector search: FAISS
- Sparse retrieval: TF-IDF

### 2. Hybrid Retrieval

Each user question is processed through two retrieval methods:

**TF-IDF retrieval:** Identifies emails containing relevant keywords
and terminology.

**Dense retrieval:** Uses E5 embeddings to identify semantically related
chunks, even when the wording differs from the question.

The two result sets are combined using Reciprocal Rank Fusion.

This allows the system to benefit from both lexical matching
and semantic similarity.

### 3. Cross-Encoder Reranking

The highest-ranked candidates from hybrid retrieval are passed to
BAAI/bge-reranker-base.

The cross-encoder evaluates the relationship between the user's question
and each candidate passage, producing a refined ranking.

The final system reranks 20 candidates before selecting the evidence
provided to the generation model.

### 4. Answer Generation

The most relevant retrieved chunks are passed to
Qwen/Qwen3-4B-Instruct-2507.

The model generates an answer using the retrieved evidence rather than
relying solely on its pretrained knowledge.

The Streamlit interface displays the generated response and provides
access to cited source emails so users can inspect the underlying evidence.

All language-model inference is performed locally.

## Evaluation Results

The system was evaluated in two stages: retrieval performance and
answer generation quality.

### Retrieval Performance

Retrieval was evaluated using 250 labelled questions, each associated
with a source email containing the expected information.

| Retrieval Method | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|
| TF-IDF | 20.0% | 42.8% | 52.8% | 0.306 |
| Dense E5 | 26.4% | 55.2% | 62.8% | 0.389 |
| Hybrid RRF | 37.2% | 67.6% | 73.6% | 0.502 |
| Hybrid RRF + BGE | **47.6%** | **73.6%** | **78.8%** | **0.588** |

Recall@K measures whether the labelled source email appears within
the top K retrieved results.

Mean Reciprocal Rank (MRR) measures how highly the correct email
is ranked across the evaluation questions.

The results show that combining sparse and dense retrieval improved
performance over either approach individually. Cross-encoder reranking
provided a further improvement.

The final pipeline achieved 78.8% Recall@10 and 0.588 MRR.

### Generation Performance

Answer generation was evaluated using 50 questions.

The base Qwen model was compared against an experimentally
fine-tuned LoRA version.

| Metric | Base Qwen | LoRA |
|---|---:|---:|
| Answer Accuracy | **76.0%** | 46.0% |
| Citation Validity | **92.0%** | 82.0% |
| Faithfulness | **82.6%** | 73.2% |

Answer accuracy measures whether the generated response contains
the information required to answer the question.

Citation validity measures whether the cited source belongs to
the retrieved evidence.

Faithfulness measures whether the generated answer is supported by
its cited evidence, considering responses with valid citations.

The base model outperformed the LoRA configuration in this evaluation.

One possible explanation is the difference between the synthetic
single-email training examples and the multi-document context
encountered during RAG inference.

The final application therefore uses the base Qwen model.

### Evaluation Limitations

These results are specific to the Enron dataset and project evaluation
questions. They should not be interpreted as general benchmarks.

Generation accuracy and faithfulness were assessed using AI-assigned
scores rather than independent human evaluation. The full judge
configuration was not preserved.

Faithfulness percentages use a different denominator from answer
accuracy because responses without valid citations were excluded.

Additional information is available in
[Evaluation Details](results/README.md).

## Technologies

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch |
| Language Model | Qwen3-4B |
| Embeddings | E5-base-v2, Sentence Transformers |
| Sparse Retrieval | TF-IDF, scikit-learn |
| Vector Search | FAISS |
| Reranking | BGE Cross-Encoder |
| Fine-Tuning Experiments | LoRA |
| Interface | Streamlit |
| Data Processing | Pandas, NumPy |

## Repository Structure

```text
enron-archive-rag/
├── README.md
├── app.py
├── generation_final.py
├── retrieval_final.py
├── requirements.txt
├── requirements-research.txt
├── .gitignore
│
├── .streamlit/
│   └── config.toml
│
├── notebooks/
│   ├── 01_data_preparation/
│   ├── 02_chunking_embeddings/
│   ├── 03_retrieval/
│   ├── 04_generation/
│   └── 05_fine_tuning/
│
├── data/
├── results/
├── docs/
└── scripts/
    └── check_assets.py
```

The research notebooks document the experiments used to develop
and evaluate the final pipeline.

See the [Notebook Guide](notebooks/README.md) for an explanation
of each stage.

## Running the Application

The application was developed for macOS using Apple Silicon.

An Apple Silicon Mac with PyTorch MPS acceleration is recommended.
A machine with 32 GB or more unified memory is a sensible starting
point, although this is not a formally tested minimum requirement.

Windows and Linux have not been tested.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/enron-archive-rag-github.git
cd enron-archive-rag-github
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a Python Environment

Using Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Python 3.12 is the suggested version, but the dependency environment
has not yet been verified from a clean installation.

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Prepare the Required Assets

The complete Enron dataset, precomputed embeddings, search indexes
and model artifacts are excluded from GitHub due to their size.

These assets must be restored or regenerated before running
the application.

See [Local Assets](docs/LOCAL_ASSETS.md) for the required files
and directory locations.

To check which assets are missing:

```bash
python scripts/check_assets.py
```

### 5. Launch the Application

```bash
python -m streamlit run app.py
```

Open the local address displayed by Streamlit.

The first run may require downloading the pretrained models.
Once downloaded, inference runs locally.

The setup has not yet been independently verified from a fresh clone.
Saved TF-IDF artifacts may also require compatible scikit-learn versions.

## Research and Experiments

The repository includes research notebooks documenting the development
of the RAG pipeline.

The experiments cover:

1. Dataset preparation and preprocessing.
2. Chunk size and overlap optimisation.
3. Embedding model comparison.
4. Sparse and dense retrieval evaluation.
5. Hybrid retrieval and reciprocal-rank fusion.
6. Cross-encoder reranking.
7. Local LLM generation and evaluation.
8. LoRA fine-tuning experiments.

These notebooks preserve the experimental methodology and
design decisions behind the final system.

To install the additional research dependencies:

```bash
python -m pip install -r requirements-research.txt
```

See [Research Notebooks](notebooks/README.md) for further details.

## Limitations and Future Improvements

Several limitations remain:

- Retrieval performance depends on the quality of the indexed evidence.
- The generation evaluation uses a relatively small question set.
- AI-based evaluation cannot replace comprehensive human assessment.
- LoRA fine-tuning did not improve performance in the tested configuration.
- The full application requires substantial local computing resources.
- Large datasets and precomputed indexes are not included in the repository.

Potential improvements include further retrieval optimisation,
larger-scale human evaluation and more efficient local inference.

## Project Context

Developed as part of the MSc Data Science and Artificial Intelligence
programme at Queen's University Belfast.

The project explores practical approaches to building, evaluating
and improving retrieval-augmented generation systems over
large collections of unstructured documents.