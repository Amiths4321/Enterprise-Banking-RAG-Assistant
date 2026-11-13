
import json


ROUTES = [
    "document_rag",
    "structured_data",
    "financial_calculation",
    "external_knowledge",
    "clarification"
]


def classify_query(client, model, question):

    prompt = f"""
Classify the following banking question.

Available routes:

1. document_rag:
   Questions about policies, procedures, guidelines,
   eligibility, and document contents.

2. structured_data:
   Questions about specific figures or information
   extracted from financial statements or records.

3. financial_calculation:
   Questions requiring mathematical financial calculations,
   such as EMI, interest, ratios, or percentages.

4. external_knowledge:
   Questions explicitly requiring current external information,
   such as the latest regulatory circular.

5. clarification:
   Questions that are too ambiguous to route safely.

Return ONLY valid JSON:

{{
  "route": "one_available_route",
  "reason": "short explanation",
  "confidence": 0.0
}}

Question:
{question}
"""

    response = client.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a banking query classification system. Return valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json",
        options={"temperature": 0}
    )

    result = json.loads(response.message.content)

    route = result.get("route")
    confidence = result.get("confidence")

    if route not in ROUTES:
        raise ValueError("Invalid route returned by model.")

    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise ValueError("Invalid routing confidence.")

    if not 0 <= confidence <= 1:
        raise ValueError("Routing confidence must be between 0 and 1.")

    if confidence < 0.70:
        result["route"] = "clarification"

    return result
