
from security_layer import (
    check_permission,
    validate_tool_arguments,
)

print("\nTEST 1: Viewer permissions")

assert check_permission("viewer", "search_documents")
assert not check_permission("viewer", "calculate_emi")

print("PASS: Viewer permissions enforced")


print("\nTEST 2: Analyst permissions")

assert check_permission("analyst", "calculate_emi")

print("PASS: Analyst permissions enforced")


print("\nTEST 3: Valid EMI inputs")

assert validate_tool_arguments(
    "calculate_emi",
    {
        "principal": 500000,
        "annual_interest_rate": 10,
        "tenure_years": 5,
    },
)

print("PASS: Valid EMI inputs accepted")


print("\nTEST 4: Invalid EMI inputs")

try:
    validate_tool_arguments(
        "calculate_emi",
        {
            "principal": -500000,
            "annual_interest_rate": 10,
            "tenure_years": 5,
        },
    )
    raise AssertionError("Invalid principal was accepted")

except ValueError:
    print("PASS: Invalid principal rejected")


print("\nTEST 5: Unknown tool")

try:
    validate_tool_arguments("delete_bank_account", {})
    raise AssertionError("Unknown tool was accepted")

except ValueError:
    print("PASS: Unknown tool rejected")


print("\nALL SECURITY TESTS COMPLETED")
