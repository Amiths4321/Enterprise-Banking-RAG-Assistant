
import sqlite3
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path


AUDIT_DB = Path(__file__).with_name("security_audit.db")

# Development identity. Replace with authenticated identity
# when integrating enterprise authentication.
DEVELOPMENT_USER = "local-dev"


# Tool permissions
ROLE_PERMISSIONS = {
    "viewer": {
        "search_documents",
        "list_banking_documents",
    },
    "analyst": {
        "search_documents",
        "list_banking_documents",
        "calculate_emi",
    },
    "admin": {
        "search_documents",
        "list_banking_documents",
        "calculate_emi",
    },
}


def initialise_audit_database():
    with sqlite3.connect(AUDIT_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                argument_summary TEXT,
                status TEXT NOT NULL,
                duration_ms REAL,
                error_message TEXT
            )
        """)


def check_permission(role, tool_name):
    allowed_tools = ROLE_PERMISSIONS.get(role, set())
    return tool_name in allowed_tools


def validate_tool_arguments(tool_name, arguments):

    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a dictionary.")

    if tool_name == "calculate_emi":
        required = {
            "principal",
            "annual_interest_rate",
            "tenure_years",
        }

        if set(arguments) != required:
            raise ValueError("Invalid EMI arguments.")

        principal = arguments["principal"]
        rate = arguments["annual_interest_rate"]
        tenure = arguments["tenure_years"]

        # Reject booleans, strings and other unexpected types.
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in (principal, rate, tenure)
        ):
            raise ValueError("EMI inputs must be numeric.")

        if not all(
            __import__("math").isfinite(value)
            for value in (principal, rate, tenure)
        ):
            raise ValueError("EMI inputs must be finite numbers.")

        if principal <= 0:
            raise ValueError("Principal must be greater than zero.")

        if not 0 <= rate <= 100:
            raise ValueError("Interest rate must be between 0 and 100.")

        if not 0 < tenure <= 50:
            raise ValueError("Tenure must be greater than 0 and at most 50 years.")

    elif tool_name == "search_documents":
        if set(arguments) != {"query"}:
            raise ValueError("Invalid document search arguments.")

        query = arguments["query"]

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if len(query) > 2000:
            raise ValueError("Search query is too long.")

    elif tool_name == "list_banking_documents":
        if arguments:
            raise ValueError("This tool does not accept arguments.")

    else:
        raise ValueError("Tool is not approved.")

    return True


def safe_argument_summary(tool_name, arguments):
    # Avoid storing raw banking questions or sensitive information.
    if tool_name == "calculate_emi":
        return json.dumps({
            "principal": arguments.get("principal"),
            "annual_interest_rate": arguments.get("annual_interest_rate"),
            "tenure_years": arguments.get("tenure_years"),
        })

    if tool_name == "search_documents":
        query = arguments.get("query", "")
        digest = hashlib.sha256(
            query.encode("utf-8")
        ).hexdigest()

        return json.dumps({
            "query_sha256": digest,
            "query_length": len(query),
        })

    return "{}"


def record_audit(
    user_id,
    role,
    tool_name,
    arguments,
    status,
    duration_ms=None,
    error_message=None,
):
    summary = safe_argument_summary(tool_name, arguments)

    with sqlite3.connect(AUDIT_DB) as conn:
        conn.execute("""
            INSERT INTO tool_audit (
                timestamp,
                user_id,
                role,
                tool_name,
                argument_summary,
                status,
                duration_ms,
                error_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            user_id,
            role,
            tool_name,
            summary,
            status,
            duration_ms,
            error_message,
        ))


initialise_audit_database()
