
from document_access import (
    grant_existing_documents_to_user,
    get_accessible_document_ids,
)

USER_ID = "local-dev"

grant_existing_documents_to_user(USER_ID)

document_ids = get_accessible_document_ids(USER_ID)

print("Document access initialised.")
print("User:", USER_ID)
print("Accessible documents:", len(document_ids))
