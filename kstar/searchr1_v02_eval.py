"""Frozen answer metrics for Search-R1 v0.2 reproduction gates."""

from __future__ import annotations

import math
import re
import string
from typing import Iterable, Sequence


def normalize_answer(text: str) -> str:
    """Exact copy of Search-R1 v0.2 ``qa_em.normalize_answer`` semantics."""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        excluded = set(string.punctuation)
        return "".join(character for character in value if character not in excluded)

    return white_space_fix(remove_articles(remove_punc(str(text).lower())))


def official_em(prediction: str | None, golden_answers: str | Sequence[str]) -> int:
    """Return the official normalized exact-match score."""
    if prediction is None:
        return 0
    answers = [golden_answers] if isinstance(golden_answers, str) else list(golden_answers)
    normalized_prediction = normalize_answer(prediction)
    return int(
        any(normalize_answer(golden_answer) == normalized_prediction for golden_answer in answers)
    )


def strict_em(prediction: str | None, golden_answers: str | Sequence[str]) -> int:
    """Case- and punctuation-sensitive exact match after outer whitespace trim."""
    if prediction is None:
        return 0
    answers = [golden_answers] if isinstance(golden_answers, str) else list(golden_answers)
    candidate = str(prediction).strip()
    return int(any(candidate == str(golden_answer).strip() for golden_answer in answers))


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Wilson score interval for a Bernoulli proportion."""
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise ValueError("successes must be an integer")
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise ValueError("total must be a positive integer")
    if successes < 0 or successes > total:
        raise ValueError("successes must be between zero and total")
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z2 / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def mean_binary(values: Iterable[int]) -> float:
    normalized = list(values)
    if not normalized:
        raise ValueError("values must not be empty")
    if any(value not in {0, 1} for value in normalized):
        raise ValueError("values must contain only zero or one")
    return sum(normalized) / len(normalized)
