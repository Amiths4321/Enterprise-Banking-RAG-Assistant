
import sqlite3
import json
from pathlib import Path


DB_PATH = Path(__file__).parent / "enterprise_documents.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialise_database():
    """Create persistent document and chunk tables."""

    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                document_name TEXT NOT NULL,
                file_size_bytes INTEGER,
                page_count INTEGER,
                chunk_count INTEGER,
                ingested_at TEXT,
                status TEXT DEFAULT 'active'
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                document_name TEXT NOT NULL,
                page_number INTEGER,
                chunk_index INTEGER,
                text TEXT NOT NULL,
                FOREIGN KEY (document_id)
                    REFERENCES documents(document_id)
                    ON DELETE CASCADE
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_document
            ON document_chunks(document_id)
        """)

        connection.commit()


def save_document(document):
    """Save document metadata and chunks.

    Re-ingesting the same file hash updates its existing record.
    """

    with get_connection() as connection:
        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        connection.execute("""
            INSERT INTO documents (
                document_id,
                document_name,
                file_size_bytes,
                page_count,
                chunk_count,
                ingested_at,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, 'active')
            ON CONFLICT(document_id) DO UPDATE SET
                document_name = excluded.document_name,
                file_size_bytes = excluded.file_size_bytes,
                page_count = excluded.page_count,
                chunk_count = excluded.chunk_count,
                ingested_at = excluded.ingested_at,
                status = 'active'
        """, (
            document["document_id"],
            document["document_name"],
            document["file_size_bytes"],
            document["page_count"],
            document["chunk_count"],
            document["ingested_at"]
        ))

        # Replace old chunks for this exact document.
        connection.execute("""
            DELETE FROM document_chunks
            WHERE document_id = ?
        """, (document["document_id"],))

        for chunk in document["chunks"]:
            connection.execute("""
                INSERT INTO document_chunks (
                    chunk_id,
                    document_id,
                    document_name,
                    page_number,
                    chunk_index,
                    text
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                chunk["chunk_id"],
                chunk["document_id"],
                chunk["document_name"],
                chunk["page_number"],
                chunk["chunk_index"],
                chunk["text"]
            ))

        connection.commit()


def list_documents():
    """Return all active document metadata."""

    with get_connection() as connection:
        rows = connection.execute("""
            SELECT *
            FROM documents
            WHERE status = 'active'
            ORDER BY ingested_at DESC
        """).fetchall()

        return [dict(row) for row in rows]


def get_document_chunks(document_id):
    """Retrieve all chunks belonging to a document."""

    with get_connection() as connection:
        rows = connection.execute("""
            SELECT *
            FROM document_chunks
            WHERE document_id = ?
            ORDER BY chunk_index
        """, (document_id,)).fetchall()

        return [dict(row) for row in rows]


def get_document_count():
    with get_connection() as connection:
        row = connection.execute("""
            SELECT COUNT(*) AS total
            FROM documents
            WHERE status = 'active'
        """).fetchone()

        return row["total"]


def delete_document(document_id):
    """Delete a document and its chunks."""

    with get_connection() as connection:
        connection.execute("PRAGMA foreign_keys = ON")

        connection.execute("""
            DELETE FROM document_chunks
            WHERE document_id = ?
        """, (document_id,))

        connection.execute("""
            DELETE FROM documents
            WHERE document_id = ?
        """, (document_id,))

        connection.commit()
