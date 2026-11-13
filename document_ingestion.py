
import hashlib
import re
from datetime import datetime, timezone

import fitz


def calculate_file_hash(file_bytes):
    """Generate a unique SHA-256 fingerprint for a document."""
    return hashlib.sha256(file_bytes).hexdigest()


def extract_pdf_pages(file_bytes):
    """Extract text from each PDF page while preserving page numbers."""
    pages = []

    with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text")

            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text.strip()

            if text:
                pages.append({
                    "page_number": page_number,
                    "text": text
                })

    return pages


def create_chunks(pages, document_id, document_name,
                  chunk_size=1200, overlap=200):
    """Split page text into overlapping chunks with source metadata."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size.")

    chunks = []
    chunk_index = 0

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": f"{document_id}_{chunk_index}",
                    "document_id": document_id,
                    "document_name": document_name,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "text": chunk_text
                })

                chunk_index += 1

            if end >= len(text):
                break

            start = end - overlap

    return chunks


def ingest_pdf(uploaded_file, chunk_size=1200, overlap=200):
    """Process one uploaded PDF and return its document record."""

    file_bytes = uploaded_file.getvalue()
    document_name = uploaded_file.name

    if not document_name.lower().endswith(".pdf"):
        raise ValueError("Only PDF documents are supported.")

    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    document_id = calculate_file_hash(file_bytes)

    pages = extract_pdf_pages(file_bytes)

    if not pages:
        raise ValueError(
            "No extractable text found. This may be a scanned PDF. "
            "OCR support will be added in a later milestone."
        )

    chunks = create_chunks(
        pages=pages,
        document_id=document_id,
        document_name=document_name,
        chunk_size=chunk_size,
        overlap=overlap
    )

    return {
        "document_id": document_id,
        "document_name": document_name,
        "file_size_bytes": len(file_bytes),
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "pages": pages,
        "chunks": chunks
    }


def ingest_multiple_pdfs(uploaded_files):
    """Ingest multiple PDFs independently."""

    results = []

    for uploaded_file in uploaded_files:
        try:
            result = ingest_pdf(uploaded_file)
            results.append({
                "success": True,
                "document": result,
                "error": None
            })

        except Exception as error:
            results.append({
                "success": False,
                "document": None,
                "file_name": uploaded_file.name,
                "error": str(error)
            })

    return results
