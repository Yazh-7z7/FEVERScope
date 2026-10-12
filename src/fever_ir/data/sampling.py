"""Deterministic sampling of claims and distractor pages.

Follows Single Responsibility Principle (SRP):
Responsible solely for deterministic selection of evaluation claims,
tuning claims, and distractor Wikipedia pages.

Uses hashlib (never Python's salted hash()) to guarantee reproducible sampling
across processes and platforms (§0 Rule 9).
"""

import hashlib
import random
from typing import Sequence

from fever_ir.data.loaders import ClaimRecord


def deterministic_page_bucket(page_title: str, modulus: int = 1000) -> int:
    """Compute a deterministic integer bucket for a page title using MD5."""
    digest = hashlib.md5(page_title.encode("utf-8")).hexdigest()
    return int(digest, 16) % modulus


def sample_claims(
    claims: Sequence[ClaimRecord],
    n_supports: int,
    n_refutes: int,
    seed: int,
) -> list[ClaimRecord]:
    """Sample stratified claims deterministically.

    Args:
        claims: Sequence of ClaimRecord candidates.
        n_supports: Target count of SUPPORTS claims.
        n_refutes: Target count of REFUTES claims.
        seed: Random seed for shuffling.

    Returns:
        List of sampled ClaimRecord objects.
    """
    rng = random.Random(seed)

    supports = [c for c in claims if c.label == "SUPPORTS"]
    refutes = [c for c in claims if c.label == "REFUTES"]

    rng.shuffle(supports)
    rng.shuffle(refutes)

    sampled = supports[:n_supports] + refutes[:n_refutes]
    rng.shuffle(sampled)
    return sampled


def select_distractor_pages(
    all_page_titles: Sequence[str],
    gold_page_titles: set[str],
    threshold: int,
    modulus: int = 1000,
) -> list[str]:
    """Select distractor pages deterministically using MD5 bucket filtering.

    A page is retained if it is a distractor and its MD5 hash % modulus < threshold.
    """
    distractors: list[str] = []
    for title in all_page_titles:
        if title not in gold_page_titles:
            if deterministic_page_bucket(title, modulus=modulus) < threshold:
                distractors.append(title)
    return distractors
