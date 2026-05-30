"""
embedder.py — Sentence Transformer encoding and semantic search
Model: all-MiniLM-L6-v2 (384-dimensional dense vectors)

WHY sentence transformers over plain BERT:
BERT produces token-level embeddings (one vector per token).
To get a sentence embedding you'd have to average token vectors — lossy.
Sentence transformers are trained with contrastive loss (siamese network):
similar sentences pulled close together in vector space,
different sentences pushed apart.
Result: one high-quality vector per sentence/chunk.

WHY all-MiniLM-L6-v2:
384 dimensions (small = fast), strong semantic quality,
best speed/quality tradeoff for deployed apps on free tier.

WHY cosine similarity not euclidean distance:
Text embeddings live on a unit hypersphere — all vectors have similar magnitude.
What matters is the ANGLE between vectors (direction = meaning).
Cosine measures exactly that. Euclidean measures raw distance — misleading here.
"""

from sentence_transformers import SentenceTransformer  # sentence transformer library
import numpy as np  # for cosine similarity computation


# load model once at module level — same reason as ner_pipeline.py
# downloading ~80MB on first run, cached locally after that
embedder_model = SentenceTransformer("all-MiniLM-L6-v2")


def encode_chunks(chunks: list[str]) -> np.ndarray:
    """
    Encode a list of text chunks into dense vectors.
    Returns a 2D numpy array of shape (num_chunks, 384).
    Each row is the embedding vector for one chunk.

    WHY encode at load time not at query time:
    Chunks don't change once document is uploaded.
    Encoding all chunks once upfront = fast search at query time.
    """
    # encode() returns numpy array by default
    # show_progress_bar=False keeps output clean in production
    embeddings = embedder_model.encode(chunks, show_progress_bar=False)
    return embeddings  # shape: (num_chunks, 384)


def encode_query(query: str) -> np.ndarray:
    """
    Encode a single search query into a dense vector.
    Returns 1D numpy array of shape (384,).
    Same model as chunks — so query and chunks live in same vector space.
    """
    # encode single string — returns shape (384,)
    return embedder_model.encode(query, show_progress_bar=False)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    Formula: cos(θ) = (A · B) / (||A|| × ||B||)
    Range: -1 (opposite) to 1 (identical direction)
    For text: typically 0.3 to 1.0

    WHY we implement this manually instead of sklearn:
    Keeps dependency count low. numpy dot product is fast enough.
    """
    # dot product of the two vectors
    dot_product = np.dot(vec_a, vec_b)

    # magnitude (L2 norm) of each vector
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    # avoid division by zero for empty/zero vectors
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def semantic_search(
    query: str,
    chunks: list[str],
    chunk_embeddings: np.ndarray,
    top_k: int = 3
) -> list[dict]:
    """
    Find the top_k most semantically relevant chunks for a query.

    Process:
    1. Encode query into same vector space as chunks
    2. Compute cosine similarity between query and every chunk
    3. Return top_k chunks sorted by similarity score

    WHY this finds "revenue" when you query "earnings":
    Both words appear in similar contexts in training data.
    Their vectors point in similar directions in the 384-dim space.
    Cosine similarity catches this — keyword search would miss it.
    """
    # encode the query into a 384-dim vector
    query_embedding = encode_query(query)

    # compute similarity between query and every chunk embedding
    similarities = []
    for i, chunk_emb in enumerate(chunk_embeddings):
        score = cosine_similarity(query_embedding, chunk_emb)
        similarities.append((i, score))

    # sort by similarity score descending — highest similarity first
    similarities.sort(key=lambda x: x[1], reverse=True)

    # take top_k results
    results = []
    for idx, score in similarities[:top_k]:
        results.append({
            "chunk": chunks[idx],        # the actual text chunk
            "score": round(score, 4),    # similarity score rounded to 4 decimal places
            "chunk_index": idx           # position in original document
        })

    return results
