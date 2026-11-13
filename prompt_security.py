
import re


INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?system\s+instructions",
    r"reveal\s+(all\s+)?(customer|account|banking)\s+information",
    r"disregard\s+(all\s+)?previous\s+instructions",
    r"you\s+are\s+now\s+the\s+system",
    r"override\s+your\s+instructions",
    r"reveal\s+your\s+system\s+prompt",
]


def detect_prompt_injection(text):

    if not isinstance(text, str):
        return {
            "is_suspicious": True,
            "matches": ["Invalid text input"],
        }

    matches = []

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            matches.append(pattern)

    return {
        "is_suspicious": bool(matches),
        "matches": matches,
    }


def sanitise_retrieved_text(text):

    result = detect_prompt_injection(text)

    if result["is_suspicious"]:
        return {
            "text": text,
            "warning": "Suspicious instruction-like content detected.",
            "is_suspicious": True,
        }

    return {
        "text": text,
        "warning": None,
        "is_suspicious": False,
    }


def build_secure_context(retrieved_chunks):

    safe_chunks = []
    suspicious_count = 0

    for chunk in retrieved_chunks:

        result = sanitise_retrieved_text(
            chunk.get("text", "")
        )

        if result["is_suspicious"]:
            suspicious_count += 1

        safe_chunks.append({
            "document_name": chunk.get("document_name"),
            "page_number": chunk.get("page_number"),
            "text": result["text"],
            "is_suspicious": result["is_suspicious"],
            "warning": result["warning"],
        })

    return {
        "chunks": safe_chunks,
        "suspicious_chunks": suspicious_count,
    }
