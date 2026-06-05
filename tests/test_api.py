"""
test_api.py — Pytest tests for FastAPI endpoints
5 tests covering: valid input, invalid input, empty input, search, health

WHY test the API not the functions directly:
API tests catch integration issues — wrong endpoint path, wrong status codes,
Pydantic validation failures. Unit tests would miss these.

Run with: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient  # TestClient lets us test FastAPI without running a server
import io

from api.main import app  # import our FastAPI app instance

# create test client — simulates HTTP requests to the app
client = TestClient(app)


# ===================== TEST 1 — Health check =====================
def test_health_check():
    """
    /health should always return 200 with status=ok.
    If this fails, the whole app is broken.
    """
    response = client.get("/health")

    # assert HTTP status code is 200
    print(response.status_code)
    print(response.json())
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "message" in data


# ===================== TEST 2 — Valid text input =====================
def test_process_text_valid():
    """
    /process-text with valid text should return 200
    and contain all expected fields in response.
    """
    sample_text = """
    Tesla Inc reported record revenues of $25 billion in Q3 2024.
    CEO Elon Musk announced expansion plans in Austin, Texas.
    The company's Model 3 continues to be the best-selling electric vehicle globally.
    Production targets for 2025 have been set at 2 million vehicles.
    The board of directors approved a new $5 billion investment in AI research.
    """

    response = client.post(
        "/process-text",
        json={"text": sample_text}
    )

    assert response.status_code == 200

    data = response.json()

    # check all expected top-level fields are present
    assert "classification" in data
    assert "entities" in data
    assert "summary" in data
    assert "chunks" in data
    assert "chunk_embeddings" in data
    assert "chunk_count" in data

    # check classification has correct structure
    assert "label" in data["classification"]
    assert "confidence" in data["classification"]
    assert 0.0 <= data["classification"]["confidence"] <= 1.0

    # check at least one chunk was created
    assert data["chunk_count"] > 0
    assert len(data["chunks"]) > 0


# ===================== TEST 3 — Empty text input =====================
def test_process_text_empty():
    """
    /process-text with empty string should return 400.
    Validates that our input guard works correctly.
    """
    response = client.post(
        "/process-text",
        json={"text": "   "}  # whitespace only = empty
    )

    # 400 = Bad Request — we explicitly raise this for empty input
    assert response.status_code == 400
    assert "detail" in response.json()


# ===================== TEST 4 — Invalid file type =====================
def test_process_pdf_invalid_format():
    """
    /process with a non-PDF file should return 422.
    Tests that our file type validation works.
    """
    # create a fake text file in memory
    fake_file = io.BytesIO(b"This is not a PDF file")

    response = client.post(
        "/process",
        files={"file": ("document.txt", fake_file, "text/plain")}  # .txt not .pdf
    )

    # 422 = Unprocessable Entity — wrong file type
    assert response.status_code == 422


# ===================== TEST 5 — Semantic search returns results =====================
def test_search_returns_results():
    """
    /search with valid query and document data should return results.
    Tests the full search pipeline end-to-end.
    """
    # first process a document to get chunks and embeddings
    sample_text = """
    Apple Inc is a technology company headquartered in Cupertino, California.
    The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.
    Apple designs, develops, and sells consumer electronics, software, and online services.
    The iPhone is Apple's flagship product and accounts for the majority of its revenue.
    In 2023, Apple became the first company to reach a market cap of 3 trillion dollars.
    Tim Cook has served as CEO since 2011, following Steve Jobs departure due to illness.
    """

    process_response = client.post("/process-text", json={"text": sample_text})
    assert process_response.status_code == 200

    doc_data = process_response.json()

    # now search the processed document
    search_response = client.post(
        "/search",
        json={
            "query": "Who founded Apple?",
            "chunks": doc_data["chunks"],
            "chunk_embeddings": doc_data["chunk_embeddings"]
        }
    )

    assert search_response.status_code == 200

    search_data = search_response.json()

    # check response structure
    assert "query" in search_data
    assert "results" in search_data
    assert len(search_data["results"]) > 0  # at least one result returned

    # check each result has required fields
    first_result = search_data["results"][0]
    assert "chunk" in first_result
    assert "score" in first_result
    assert "chunk_index" in first_result
    assert 0.0 <= first_result["score"] <= 1.0  # cosine similarity range
