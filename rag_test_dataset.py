
TEST_CASES = [
    {
        "question": "What was the closing balance?",
        "relevant_chunk_ids": ["chunk_001"],
        "expected_answer": "INR 68450.25",
        "expected_sources": [
            {
                "document_name": "Sample Bank Statement",
                "page_number": 1,
            }
        ],
    },
    {
        "question": "What documents are required for self-employed applicants?",
        "relevant_chunk_ids": ["chunk_002", "chunk_003"],
        "expected_answer": "Income tax returns and bank statements",
        "expected_sources": [
            {
                "document_name": "Sample Loan Policy",
                "page_number": 2,
            }
        ],
    },
]
