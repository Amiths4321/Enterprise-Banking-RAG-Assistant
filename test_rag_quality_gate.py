
from rag_quality_gate import validate_rag_response


valid_result = {
    "answer": "The closing balance was ₹68,450.25.",
    "sources": [
        {
            "document_name": "PDF Bank Statement - HDFC Bank.pdf",
            "page_number": 1,
        }
    ],
}

invalid_result = {
    "answer": "The closing balance was ₹68,450.25.",
    "sources": [],
}

print("VALID RESPONSE:")
print(validate_rag_response(valid_result))

print("\nINVALID RESPONSE:")
print(validate_rag_response(invalid_result))
