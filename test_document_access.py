
from document_access import (
    grant_document_access,
    revoke_document_access,
    user_can_access_document,
)

doc_a = "DEMO_CUSTOMER_A"
doc_b = "DEMO_CUSTOMER_B"

user_a = "customer_a"
user_b = "customer_b"

# Assign separate documents.
grant_document_access(doc_a, user_a)
grant_document_access(doc_b, user_b)

# Positive tests.
assert user_can_access_document(user_a, doc_a)
assert user_can_access_document(user_b, doc_b)

# Negative tests.
assert not user_can_access_document(user_a, doc_b)
assert not user_can_access_document(user_b, doc_a)

print("PASS: Customer A can access their document.")
print("PASS: Customer B can access their document.")
print("PASS: Customer A cannot access Customer B's document.")
print("PASS: Customer B cannot access Customer A's document.")

# Clean up demonstration entries.
revoke_document_access(doc_a, user_a)
revoke_document_access(doc_b, user_b)

print("\nAll document access tests passed.")
