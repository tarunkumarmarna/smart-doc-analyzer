"""
main.py — FastAPI application with 3 endpoints
/process  POST — full document analysis (classification + NER + summary + embeddings)
/search   POST — semantic search over processed document
/health   GET  — always include in production APIs

WHY FastAPI over Flask:
FastAPI auto-generates /docs (Swagger UI) — interactive API documentation for free.
Pydantic models validate request/response automatically — no manual validation.
Type hints = self-documenting code.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File  # FastAPI core
from pydantic import BaseModel        # for request/response validation
from typing import Optional           # for optional fields
import uvicorn                        # ASGI server to run FastAPI

from src.pipeline import process_document, search_document  # our ML pipeline


# create FastAPI app instance
# title and description appear in auto-generated /docs page
app = FastAPI(
    title="Smart Document Analyzer",
    description="Upload any PDF or text. Ask questions in plain English, get answers from the document.",
    version="1.0.0"
)


# --- Pydantic Models ---
# WHY Pydantic: automatically validates incoming request data
# if field is wrong type or missing, FastAPI returns 422 error automatically

class TextRequest(BaseModel):
    """Request model for plain text input to /process"""
    text: str                          # raw text content
    filename: Optional[str] = "document"  # optional display name


class SearchRequest(BaseModel):
    """Request model for /search endpoint"""
    query: str                          # user's search question
    chunks: list[str]                   # chunks from /process response
    chunk_embeddings: list[list[float]] # embeddings from /process response


class HealthResponse(BaseModel):
    """Response model for /health endpoint"""
    status: str
    message: str


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint.
    WHY always include: Render and other platforms ping /health to check if app is alive.
    Returns 200 if app is running correctly.
    """
    return HealthResponse(status="ok", message="Smart Document Analyzer is running")


@app.post("/process")
async def process_pdf(file: UploadFile = File(...)):
    """
    Process an uploaded PDF file.
    Returns: classification, entities, summary, chunks, embeddings.

    WHY async: file upload is I/O bound — async allows other requests during upload.
    UploadFile = FastAPI's file upload type — handles multipart form data.
    File(...) = required field (... means required in FastAPI).
    """
    # validate file type — only accept PDFs
    if not file.filename.endswith(".pdf"):
        # 422 = Unprocessable Entity — client sent wrong data type
        raise HTTPException(status_code=422, detail="Only PDF files accepted. Use /process-text for plain text.")

    # read file bytes from upload
    file_bytes = await file.read()

    # guard: empty file
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        # run full ML pipeline
        result = process_document(file_bytes=file_bytes)
    except ValueError as e:
        # ValueError raised when PDF has no extractable text (scanned/image PDF)
        raise HTTPException(status_code=400, detail=str(e))

    return result


@app.post("/process-text")
def process_text(request: TextRequest):
    """
    Process plain text input (alternative to PDF upload).
    Same pipeline as /process but accepts raw text directly.
    """
    # guard: empty text
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text input is empty.")

    try:
        result = process_document(raw_text=request.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result


@app.post("/search")
def search(request: SearchRequest):
    """
    Semantic search over a previously processed document.
    Client sends back chunks + embeddings from /process response.

    WHY client sends embeddings back:
    API is stateless — no session storage between requests.
    Client holds the document state and sends it with each search query.
    This is the correct REST design pattern.
    """
    # guard: empty query
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Search query is empty.")

    # guard: no chunks to search
    if not request.chunks:
        raise HTTPException(status_code=400, detail="No document chunks provided. Process a document first.")

    result = search_document(
        query=request.query,
        chunks=request.chunks,
        chunk_embeddings_list=request.chunk_embeddings
    )

    return result


# run app locally with: python -m uvicorn api.main:app --reload
# --reload: auto-restart on code changes (development only)
if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
