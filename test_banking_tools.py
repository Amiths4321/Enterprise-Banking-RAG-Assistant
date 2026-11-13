
from banking_tools import calculate_loan_emi

result = calculate_loan_emi(
    principal=500000,
    annual_interest_rate=10,
    tenure_years=5,
)

print("EMI TEST")
print(result)

assert result["monthly_emi"] == 10623.52
assert result["currency"] == "INR"

print("\nEMI TEST PASSED")
