"""Check local runtime assets without loading models or serialized indexes."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
required = ["enron_clean.csv", "full_corpus_tfidf_matrix.npz",
            "full_corpus_tfidf_vectorizer.joblib",
            "e5_embeddings_final/embedding_manifest.json",
            "e5_embeddings_final/e5_dense_index.faiss"]
manifest_path = root / required[3]
if manifest_path.is_file():
    manifest = json.loads(manifest_path.read_text())
    required.extend(shard["chunks_file"] for shard in manifest["shards"])
missing = [name for name in required if not (root / name).is_file()]
if missing:
    print("Missing local assets:\n" + "\n".join(missing))
    raise SystemExit(1)
print(f"All {len(required)} required runtime asset files are present.")
