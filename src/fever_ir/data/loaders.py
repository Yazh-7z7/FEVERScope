"""Streaming loaders for raw FEVER claim files.

Follows Single Responsibility Principle (SRP):
Responsible solely for parsing and yielding raw claim records from JSONL files
without loading entire datasets into memory.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fever_ir.utils.io import read_jsonl_stream


@dataclass(frozen=True)
class EvidenceItem:
    annotation_id: int | None
    evidence_id: int | None
    page_title: str | None
    sentence_id: int | None


@dataclass(frozen=True)
class ClaimRecord:
    id: int
    verifiable: str
    label: str
    claim: str
    evidence_sets: list[list[EvidenceItem]]

    @property
    def is_nei(self) -> bool:
        """Return True if claim has NOT ENOUGH INFO label."""
        return self.label == "NOT ENOUGH INFO"

    @property
    def flattened_gold_sentences(self) -> set[tuple[str, int]]:
        """Return deduplicated set of (page_title, sentence_id) for valid evidence."""
        gold = set()
        for ev_set in self.evidence_sets:
            for item in ev_set:
                if item.page_title is not None and item.sentence_id is not None:
                    gold.add((item.page_title, item.sentence_id))
        return gold


def parse_claim_dict(raw: dict[str, Any]) -> ClaimRecord:
    """Parse a raw dictionary from FEVER JSONL into a typed ClaimRecord."""
    evidence_sets: list[list[EvidenceItem]] = []
    raw_evidence = raw.get("evidence", [])

    for ev_set in raw_evidence:
        current_set: list[EvidenceItem] = []
        for item in ev_set:
            if len(item) >= 4:
                ann_id, ev_id, page, sent_id = item[0], item[1], item[2], item[3]
                current_set.append(
                    EvidenceItem(
                        annotation_id=ann_id if isinstance(ann_id, int) else None,
                        evidence_id=ev_id if isinstance(ev_id, int) else None,
                        page_title=str(page) if page is not None else None,
                        sentence_id=int(sent_id) if sent_id is not None else None,
                    )
                )
        if current_set:
            evidence_sets.append(current_set)

    return ClaimRecord(
        id=int(raw["id"]),
        verifiable=str(raw.get("verifiable", "")),
        label=str(raw.get("label", "")),
        claim=str(raw.get("claim", "")),
        evidence_sets=evidence_sets,
    )


def load_claims_stream(path: Path | str) -> Iterator[ClaimRecord]:
    """Stream parsed ClaimRecord objects from a FEVER JSONL file."""
    for record_dict in read_jsonl_stream(path):
        yield parse_claim_dict(record_dict)
