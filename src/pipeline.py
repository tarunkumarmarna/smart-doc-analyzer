"""
pipeline.py — Master pipeline connecting all components
This is what main.py calls. Single entry point for all ML logic.

WHY this file exists:
main.py should not know about PyPDF2, NER, embeddings separately.
It calls process_document() and search_document() — that's it.
Clean separation: API layer knows nothing about ML internals.
"""

from src.extractor import process_input          # PDF/text extraction + chunking
from src.ner_pipeline import extract_entities_from_chunks  # NER across chunks
from src.embedder import encode_chunks, semantic_search    # embeddings + search
from src.summarizer import summarize_text        # abstractive summarization
from src.classifier import classify_document     # distilBERT classification


def process_document(file_bytes: bytes = None, raw_text: str = None) -> dict:
    """
    Full document analysis pipeline.
    Input: PDF bytes OR plain text string
    Output: classification + entities + summary + chunks (for search later)

    Called by: POST /process endpoint in main.py
    """
    # STEP 1: extract text and split into overlapping chunks
    full_text, chunks = process_input(file_bytes=file_bytes, raw_text=raw_text)

    # guard: if extraction returned nothing (blank/scanned PDF)
    if not full_text.strip():
        raise ValueError("Could not extract text. PDF may be scanned or image-based. Plain text input supported.")

    # STEP 2: classify full document type using fine-tuned distilBERT
    # WHY full text not chunks: classification needs global document context
    # truncation in classifier.py handles documents longer than 512 tokens
    classification = classify_document(full_text[:2000])  # first 2000 chars = enough for doc type

    # STEP 3: extract named entities across all chunks
    # WHY chunks not full text: NER model has 512 token limit per call
    entities = extract_entities_from_chunks(chunks)

    # STEP 4: summarize full document
    # WHY first 1000 words: summarization model also has token limits
    # first 1000 words captures the most important content (intro + key points)
    summary_input = " ".join(full_text.split()[:1000])
    summary = summarize_text(summary_input)

    # STEP 5: encode all chunks into embeddings for semantic search
    # stored in response so /search endpoint can reuse without re-encoding
    # WHY return embeddings: avoids re-processing same document on search calls
    chunk_embeddings = encode_chunks(chunks)

    return {
        "full_text": full_text,                         # raw extracted text
        "chunks": chunks,                               # list of text chunks
        "chunk_embeddings": chunk_embeddings.tolist(),  # embeddings as list (JSON serializable)
        "classification": classification,               # label + confidence + all_scores
        "entities": entities,                           # grouped by type
        "summary": summary,                             # abstractive summary string
        "chunk_count": len(chunks),                     # how many chunks extracted
    }


def search_document(query: str, chunks: list[str], chunk_embeddings_list: list) -> dict:
    """
    Semantic search over already-processed document.
    Input: query string + chunks + embeddings (from process_document output)
    Output: top-3 most relevant chunks with similarity scores

    Called by: POST /search endpoint in main.py
    WHY separate from process_document:
    User may search multiple times on same document.
    No need to re-extract, re-classify, re-encode every time.
    """
    import numpy as np

    # convert embeddings back from list to numpy array
    # WHY: JSON serialization requires list, numpy required for cosine similarity
    chunk_embeddings = np.array(chunk_embeddings_list)

    # run semantic search — returns top 3 chunks with scores
    results = semantic_search(
        query=query,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        top_k=3
    )

    return {
        "query": query,
        "results": results  # list of {chunk, score, chunk_index}
    }
