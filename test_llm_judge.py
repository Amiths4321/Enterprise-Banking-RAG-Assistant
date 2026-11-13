
from llm_judge import evaluate_answer


question = "What was the closing balance?"


evidence = """
The sample HDFC bank statement shows a closing balance
of INR 68,450.25 on page 1.
"""


# Test 1: Supported answer
supported_answer = """
The closing balance was INR 68,450.25.
"""

print("\nTEST 1: Supported answer")

result1 = evaluate_answer(
    question,
    evidence,
    supported_answer,
)

print(result1)


# Test 2: Unsupported answer
unsupported_answer = """
The closing balance was INR 95,000.
The account holder also received a bonus of INR 20,000.
"""

print("\nTEST 2: Unsupported answer")

result2 = evaluate_answer(
    question,
    evidence,
    unsupported_answer,
)

print(result2)


print("\nLLM JUDGE TESTING COMPLETED")
