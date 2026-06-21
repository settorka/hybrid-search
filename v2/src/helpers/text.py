import re
from collections import Counter
from hashlib import blake2b
from math import sqrt

TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_query(text: str) -> str:
    """Normalize user text for cache keys and matching."""

    return " ".join(tokenize(text))


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric terms."""

    return TOKEN_RE.findall(text.lower())


def hash_embedding(text: str, dimension: int) -> list[float]:
    """Build a deterministic bounded embedding for local v2 search."""

    vector = [0.0] * dimension
    for token, count in Counter(tokenize(text)).items():
        digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign * float(count)
    return normalize_vector(vector)


def normalize_vector(vector: list[float]) -> list[float]:
    """Normalize a vector to unit length when possible."""

    norm = sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for equally sized vectors."""

    if len(left) != len(right):
        raise ValueError("vector dimensions differ")
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )


def build_snippet(content: str, max_length: int = 180) -> str:
    """Return a bounded content snippet."""

    if len(content) <= max_length:
        return content
    return f"{content[: max_length - 3].rstrip()}..."
