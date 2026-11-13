
import time
import sys
from contextlib import redirect_stdout

from mcp.server.fastmcp import FastMCP

from banking_tools import (
    search_banking_documents,
    calculate_loan_emi,
    get_document_metadata,
)

from security_layer import (
    check_permission,
    validate_tool_arguments,
    record_audit,
    DEVELOPMENT_USER,
)

mcp = FastMCP("Enterprise Banking Assistant")

# Development role only.
# Never trust a role supplied directly by an AI model.
CURRENT_ROLE = "admin"


def secure_execute(tool_name, arguments, function):

    start = time.perf_counter()
    status = "success"
    error_message = None

    try:
        if not check_permission(CURRENT_ROLE, tool_name):
            raise PermissionError(
                "You are not authorised to use this tool."
            )

        validate_tool_arguments(tool_name, arguments)

        with redirect_stdout(sys.stderr):
            return function()

    except Exception as error:
        status = "failure"
        error_message = str(error)
        raise

    finally:
        duration = (time.perf_counter() - start) * 1000

        record_audit(
            user_id=DEVELOPMENT_USER,
            role=CURRENT_ROLE,
            tool_name=tool_name,
            arguments=arguments,
            status=status,
            duration_ms=round(duration, 2),
            error_message=error_message,
        )


@mcp.tool()
def search_documents(query: str) -> dict:
    """Search banking documents for relevant evidence."""

    arguments = {"query": query}

    return secure_execute(
        "search_documents",
        arguments,
        lambda: search_banking_documents(query),
    )


@mcp.tool()
def calculate_emi(principal, annual_interest_rate, tenure_years):
    """Calculate monthly loan EMI in INR."""

    principal = float(principal)
    annual_interest_rate = float(annual_interest_rate)
    tenure_years = float(tenure_years)

    if principal <= 0 or tenure_years <= 0 or annual_interest_rate < 0:
        raise ValueError(
            "Principal and tenure must be positive, and the rate cannot be negative."
        )

    n = int(round(tenure_years * 12))
    r = annual_interest_rate / 12 / 100

    if r == 0:
        emi = principal / n
    else:
        factor = (1 + r) ** n
        emi = principal * r * factor / (factor - 1)

    total_payment = emi * n

    return {
        "monthly_emi": round(emi, 2),
        "number_of_installments": n,
        "total_payment": round(total_payment, 2),
        "total_interest": round(total_payment - principal, 2),
        "currency": "INR",
    }


@mcp.tool()
def list_banking_documents() -> dict:
    """List metadata for available banking documents."""

    return secure_execute(
        "list_banking_documents",
        {},
        get_document_metadata,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")

def get_document_metadata():
    """List metadata for the banking documents the current user may access."""

    document_ids = get_accessible_document_ids()

    return {
        "documents": [{"document_id": doc_id} for doc_id in document_ids],
        "count": len(document_ids),
    }