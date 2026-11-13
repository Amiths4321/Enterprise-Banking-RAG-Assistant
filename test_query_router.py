
from ollama import Client
from query_router import classify_query

client = Client(
    host="http://10.22.39.192:11434"
)

model = "qwen3-coder:30b"

questions = [
    "What is the KYC verification procedure?",
    "What was the closing balance in my bank statement?",
    "Calculate EMI for a loan of INR 500000 at 10 percent for 5 years.",
    "What is the latest RBI circular on digital lending?",
    "Can you help me with this?"
]

for question in questions:

    result = classify_query(
        client,
        model,
        question
    )

    print("\nQuestion:", question)
    print("Route:", result["route"])
    print("Reason:", result["reason"])
    print("Confidence:", result["confidence"])
    print("-" * 60)
