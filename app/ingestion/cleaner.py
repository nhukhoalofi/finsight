import re
import unicodedata


def clean_text(text: str) -> str:
    """Normalize extraction artefacts without changing financial meaning."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = "".join(char for char in normalized if char >= " " or char in "\n\t")
    normalized = re.sub(r"[^\S\r\n]+", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()
