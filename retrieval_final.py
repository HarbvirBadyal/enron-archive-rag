# # Final hybrid retrieval pipeline
#
# Standalone retrieval pipeline: TF-IDF + E5/FAISS → reciprocal-rank fusion → BGE reranker. All runtime settings are defined directly in this notebook.
#
# The current hybrid configuration fuses the top 100 emails from each retriever and reranks 20 candidate emails, selecting one E5 chunk per email. `retrieve_chunks(question, k=5)` returns the best k chunks after reranking.


# Imports
import json
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from scipy.sparse import load_npz
from sentence_transformers import CrossEncoder, SentenceTransformer

BASE_DIRECTORY = Path.cwd()
EMAILS_FILE = BASE_DIRECTORY / 'enron_clean.csv'
EMBEDDING_DIRECTORY = BASE_DIRECTORY / 'e5_embeddings_final'
MANIFEST_FILE = EMBEDDING_DIRECTORY / 'embedding_manifest.json'
DENSE_INDEX_FILE = EMBEDDING_DIRECTORY / 'e5_dense_index.faiss'
TFIDF_VECTORIZER_FILE = BASE_DIRECTORY / 'full_corpus_tfidf_vectorizer.joblib'
TFIDF_MATRIX_FILE = BASE_DIRECTORY / 'full_corpus_tfidf_matrix.npz'

# Final retrieval settings, copied from the evaluated hybrid configuration.
EMBEDDING_MODEL_NAME = 'intfloat/e5-base-v2'
RERANKER_MODEL_NAME = 'BAAI/bge-reranker-base'
QUERY_PREFIX = 'query: '
FUSION_DEPTH = 100
INITIAL_CHUNK_FETCH = 5_000
RRF_CONSTANT = 60
LEXICAL_WEIGHT = 1.0
DENSE_WEIGHT = 1.0
RERANK_DEPTH = 20
RERANKER_BATCH_SIZE = 16

if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

print(f'Configuration ready; neural models will use {DEVICE}.')


# Load and validate the final retrieval indexes and build the mappings from chunks back to source emails.
emails = pd.read_csv(EMAILS_FILE)
emails['email_id'] = emails['email_id'].astype(str)
manifest = json.loads(MANIFEST_FILE.read_text(encoding='utf-8'))
dense_index = faiss.read_index(str(DENSE_INDEX_FILE))
tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_FILE)
email_tfidf_matrix = load_npz(TFIDF_MATRIX_FILE).tocsr()

required_columns = {'email_id', 'file', 'date', 'sender', 'recipients', 'subject', 'body'}
missing_columns = required_columns - set(emails.columns)
if missing_columns:
    raise ValueError(f'Missing email columns: {sorted(missing_columns)}')
if not manifest.get('complete') or manifest['model_name'] != EMBEDDING_MODEL_NAME:
    raise ValueError('The completed embedding manifest does not match this pipeline.')
if manifest['number_of_emails'] != len(emails):
    raise ValueError('The embedding manifest does not match the email corpus.')
if dense_index.ntotal != manifest['number_of_chunks']:
    raise ValueError('The FAISS index does not match the embedding manifest.')
if email_tfidf_matrix.shape[0] != len(emails):
    raise ValueError('The TF-IDF matrix does not match the email corpus.')

# Map every FAISS vector to its email and to its row inside a parquet shard.
email_count = len(emails)
chunk_source_rows = np.empty(dense_index.ntotal, dtype=np.int32)
email_shard = np.full(email_count, -1, dtype=np.int32)
email_local_start = np.full(email_count, -1, dtype=np.int32)
email_local_end = np.full(email_count, -1, dtype=np.int32)
shard_global_start = {}
shards_by_number = {}
global_offset = 0

for shard in manifest['shards']:
    shard_number = int(shard['shard_number'])
    shards_by_number[shard_number] = shard
    shard_global_start[shard_number] = global_offset
    source_rows = pd.read_parquet(
        BASE_DIRECTORY / shard['chunks_file'], columns=['source_row']
    )['source_row'].to_numpy(dtype=np.int64)
    next_offset = global_offset + len(source_rows)
    chunk_source_rows[global_offset:next_offset] = source_rows
    if len(source_rows):
        starts = np.r_[0, np.flatnonzero(source_rows[1:] != source_rows[:-1]) + 1]
        ends = np.r_[starts[1:], len(source_rows)]
        positions = source_rows[starts]
        email_shard[positions] = shard_number
        email_local_start[positions] = starts
        email_local_end[positions] = ends
    global_offset = next_offset

if global_offset != dense_index.ntotal:
    raise ValueError('Shard metadata does not align with the FAISS index.')

_embedding_model = None
_reranker = None
_shard_cache = {}
print(f'Indexes ready: {len(emails):,} emails and {dense_index.ntotal:,} chunks.')


# Define the lexical, dense, fusion, reranking, and source recovery functions used by final retrieval.

# Convert missing values to clean strings.
def _text(value):
    return '' if pd.isna(value) else str(value).strip()

# Load the E5 embedding model and BGE reranker only when retrieval is first used.
def _models():
    global _embedding_model, _reranker
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME, max_length=512, device=DEVICE)
    return _embedding_model, _reranker

# Retrieve the highest-scoring email candidates using TF-IDF lexical search.
def _top_lexical(question):
    query_vector = tfidf_vectorizer.transform([question])
    scores = (query_vector @ email_tfidf_matrix.T).toarray().ravel()
    positions = np.flatnonzero(scores > 0)
    if len(positions) > FUSION_DEPTH:
        selected = np.argpartition(scores[positions], -FUSION_DEPTH)[-FUSION_DEPTH:]
        positions = positions[selected]
    ranking = positions[np.argsort(scores[positions])[::-1]].astype(np.int64)
    return ranking, {int(position): float(scores[position]) for position in ranking}

# Retrieve dense E5 candidates from FAISS and collapse repeated chunks into unique emails.
def _top_dense(query_embedding):
    chunk_fetch = min(INITIAL_CHUNK_FETCH, dense_index.ntotal)
    while True:
        scores, vector_ids = dense_index.search(query_embedding[None, :], chunk_fetch)
        ranking = []
        hits = {}
        for score, vector_id in zip(scores[0], vector_ids[0]):
            if vector_id < 0:
                continue
            email_position = int(chunk_source_rows[vector_id])
            if email_position in hits:
                continue
            hits[email_position] = {'score': float(score), 'vector_id': int(vector_id)}
            ranking.append(email_position)
            if len(ranking) == FUSION_DEPTH:
                break
        if len(ranking) == FUSION_DEPTH or chunk_fetch == dense_index.ntotal:
            return np.asarray(ranking, dtype=np.int64), hits
        chunk_fetch = min(chunk_fetch * 2, dense_index.ntotal)

# Combine the lexical and dense rankings using reciprocal rank fusion.
def _fuse(lexical_ranking, dense_ranking):
    scores = {}
    for rank, position in enumerate(lexical_ranking[:FUSION_DEPTH], start=1):
        position = int(position)
        scores[position] = scores.get(position, 0.0) + LEXICAL_WEIGHT / (RRF_CONSTANT + rank)
    for rank, position in enumerate(dense_ranking[:FUSION_DEPTH], start=1):
        position = int(position)
        scores[position] = scores.get(position, 0.0) + DENSE_WEIGHT / (RRF_CONSTANT + rank)
    ranking = sorted(scores, key=lambda position: scores[position], reverse=True)
    return ranking, scores

# Load and cache the chunk metadata
def _load_shard(shard_number):
    if shard_number not in _shard_cache:
        shard = shards_by_number[shard_number]
        _shard_cache[shard_number] = pd.read_parquet(
            BASE_DIRECTORY / shard['chunks_file'],
            columns=['chunk_number', 'chunk_id', 'chunk_text'],
        )
    return _shard_cache[shard_number]

# Find the most relevant chunk within a candidate email using E5 similarity
def _best_chunk(email_position, query_embedding):
    shard_number = int(email_shard[email_position])
    start = int(email_local_start[email_position])
    end = int(email_local_end[email_position])
    if shard_number < 0 or start < 0 or end <= start:
        raise RuntimeError(f'No embedded chunk found for email row {email_position}.')
    vector_start = shard_global_start[shard_number] + start
    vectors = dense_index.reconstruct_n(vector_start, end - start)
    chunk_scores = vectors @ query_embedding
    best_row = start + int(np.argmax(chunk_scores))
    chunk = _load_shard(shard_number).iloc[best_row]
    return {
        'chunk_number': int(chunk['chunk_number']),
        'chunk_id': _text(chunk['chunk_id']),
        'chunk_text': _text(chunk['chunk_text']),
        'passage_dense_score': float(np.max(chunk_scores)),
    }

# Run the full hybrid retrieval pipeline and return the top reranked evidence chunks
def retrieve_chunks(question, k=5):
    """Return top-k chunks from hybrid retrieval followed by BGE reranking."""
    question = str(question).strip()
    if not question:
        raise ValueError('question cannot be empty.')
    if not 1 <= int(k) <= RERANK_DEPTH:
        raise ValueError(f'k must be between 1 and {RERANK_DEPTH}.')

    embedding_model, reranker = _models()
    query_embedding = embedding_model.encode(
        [QUERY_PREFIX + question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)[0]

    lexical_ranking, lexical_scores = _top_lexical(question)
    dense_ranking, dense_hits = _top_dense(query_embedding)
    hybrid_ranking, rrf_scores = _fuse(lexical_ranking, dense_ranking)
    candidates = hybrid_ranking[:RERANK_DEPTH]
    lexical_ranks = {int(position): rank for rank, position in enumerate(lexical_ranking, 1)}
    dense_ranks = {int(position): rank for rank, position in enumerate(dense_ranking, 1)}

    evidence = {position: _best_chunk(position, query_embedding) for position in candidates}
    pairs = [
        (question, f"Subject: {_text(emails.iloc[position]['subject'])}\n{evidence[position]['chunk_text']}")
        for position in candidates
    ]
    reranker_values = np.asarray(
        reranker.predict(pairs, batch_size=RERANKER_BATCH_SIZE, show_progress_bar=False)
    ).reshape(-1)
    reranker_scores = {position: float(score) for position, score in zip(candidates, reranker_values)}
    candidate_order = {position: rank for rank, position in enumerate(candidates)}
    reranked = sorted(
        candidates, key=lambda position: (-reranker_scores[position], candidate_order[position])
    )[:int(k)]

    results = []
    for rank, position in enumerate(reranked, start=1):
        email = emails.iloc[position]
        chunk = evidence[position]
        results.append({
            'rank': rank,
            'email_id': _text(email['email_id']),
            'chunk_id': chunk['chunk_id'],
            'chunk_number': chunk['chunk_number'],
            'file': _text(email['file']),
            'date': _text(email['date']),
            'sender': _text(email['sender']),
            'recipients': _text(email['recipients']),
            'subject': _text(email['subject']),
            'chunk_text': chunk['chunk_text'],
            'lexical_rank': lexical_ranks.get(position),
            'lexical_score': lexical_scores.get(position, 0.0),
            'dense_rank': dense_ranks.get(position),
            'dense_score': dense_hits.get(position, {}).get('score', np.nan),
            'rrf_score': rrf_scores[position],
            'reranker_score': reranker_scores[position],
        })
    return results

# Return nearby chunks from the same email to provide additional context around a retrieved chunk.
def get_adjacent_chunks(email_id, chunk_number, radius=1):
    """Return the selected chunk with neighbouring chunks from the same email."""
    matches = np.flatnonzero(emails['email_id'].to_numpy() == str(email_id))
    if len(matches) != 1:
        raise KeyError(f'Expected one email for {email_id!r}; found {len(matches)}.')
    email_position = int(matches[0])
    shard_number = int(email_shard[email_position])
    start = int(email_local_start[email_position])
    end = int(email_local_end[email_position])
    chunk_rows = _load_shard(shard_number).iloc[start:end]
    lower = max(0, int(chunk_number) - int(radius))
    upper = int(chunk_number) + int(radius)
    selected = chunk_rows[chunk_rows['chunk_number'].between(lower, upper)]
    return '\n'.join(selected.sort_values('chunk_number')['chunk_text'].map(_text))

# Return the complete original email and its metadata for a selected source.
def get_full_email(email_id):
    """Return the original stored fields for one retrieved email."""
    matches = emails[emails['email_id'] == str(email_id)]
    if len(matches) != 1:
        raise KeyError(f'Expected one email for {email_id!r}; found {len(matches)}.')
    email = matches.iloc[0]
    return {name: _text(email[name]) for name in ['email_id', 'date', 'sender', 'recipients', 'subject', 'body']}


print('retrieve_chunks(question, k=5) is ready.')


# Run this example when I open this notebook directly. The generation evaluation
# sets RUN_RETRIEVAL_EXAMPLE=False so it can import the pipeline without an extra query.
RUN_RETRIEVAL_EXAMPLE = globals().get('RUN_RETRIEVAL_EXAMPLE', True)
if __name__ == "__main__" and RUN_RETRIEVAL_EXAMPLE:
    question = 'What was discussed about the California energy market?'
    retrieved_chunks = retrieve_chunks(question, k=5)
    display(pd.DataFrame(retrieved_chunks))
