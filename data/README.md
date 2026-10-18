# Research Data

This directory contains the evaluation and training datasets used
during the development of the Enron Archive RAG system.

The datasets are stored locally and excluded from the public GitHub
repository because they contain information derived from the
original Enron email corpus.

## Datasets

| Dataset | Description | Purpose |
|---|---|---|
| `enron_evaluation_250.csv` | 250 labelled questions associated with source emails | Evaluate retrieval performance and support generation experiments |
| `enron_qa_training_2000.csv` | 2,000 synthetic question-answer examples derived from individual emails | Train and validate the experimental LoRA configuration |

### Evaluation Dataset

The evaluation dataset contains 250 questions associated with
source emails from the Enron archive.

It was used to evaluate retrieval performance using metrics
including Recall@K and Mean Reciprocal Rank (MRR).

A subset of 50 questions was also used for the reported
generation evaluation.

### LoRA Training Dataset

The LoRA experiment used 2,000 synthetic question-answer examples
generated from individual emails.

The dataset was divided into:

- 1,800 training examples.
- 200 validation examples.

The fine-tuning experiment investigated whether adapting the
base Qwen model to email-based question answering improved
performance within the RAG pipeline.

## Additional Data Requirements

Some research notebooks require additional datasets stored
at the repository root.

| File | Purpose |
|---|---|
| `emails.csv` | Original Enron email archive used for data preparation |
| `enron_clean.csv` | Cleaned email corpus used by the retrieval pipeline |
| `enron_stratified_sample.csv` | Intermediate dataset used in exploratory experiments |

These datasets are not included in the public repository.

The original archive must be obtained from an appropriate source
before the data preparation workflow can be reproduced.

The stratified sample must also be recreated or obtained
before running notebooks that depend on it.

## Reproducibility

The research notebooks document the experimental methodology,
but reproducing the full evaluation requires the corresponding
datasets, indexes and model dependencies.

The absence of these local datasets does not prevent readers
from examining the experimental code or reported results.

For additional information, see:

- [Research Notebooks](../notebooks/README.md)
- [Evaluation Results](../results/README.md)
- [Local Assets](../docs/LOCAL_ASSETS.md)