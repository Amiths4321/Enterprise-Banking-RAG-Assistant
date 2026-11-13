
import sqlite3
from pathlib import Path

from document_store import list_documents


DB_PATH = Path(__file__).with_name("security_audit.db")


def initialise_access_table():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_acl (
                document_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                permission TEXT NOT NULL DEFAULT 'read',
                PRIMARY KEY (document_id, user_id)
            )
        """)


def grant_document_access(document_id, user_id, permission="read"):
    if permission not in ("read", "admin"):
        raise ValueError("Invalid permission.")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO document_acl
                (document_id, user_id, permission)
            VALUES (?, ?, ?)
            ON CONFLICT(document_id, user_id)
            DO UPDATE SET permission = excluded.permission
        """, (document_id, user_id, permission))


def revoke_document_access(document_id, user_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            DELETE FROM document_acl
            WHERE document_id = ? AND user_id = ?
        """, (document_id, user_id))


def get_accessible_document_ids(user_id):
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("""
            SELECT document_id
            FROM document_acl
            WHERE user_id = ?
              AND permission IN ('read', 'admin')
        """, (user_id,)).fetchall()

    return {row[0] for row in rows}


def user_can_access_document(user_id, document_id):
    return document_id in get_accessible_document_ids(user_id)


def grant_existing_documents_to_user(user_id):
    """
    Development initialisation only.
    Grants the specified user read access to all currently
    ingested documents. Do not use this as production onboarding.
    """
    documents = list_documents()

    for document in documents:
        grant_document_access(
            document["document_id"],
            user_id,
            "read",
        )


initialise_access_table()
