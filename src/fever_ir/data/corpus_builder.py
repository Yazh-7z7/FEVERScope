"""Corpus construction and TREC qrels compilation.

Follows Single Responsibility Principle (SRP):
Builds the canonical Corpus DataFrame (single source of truth for all retrievers)
and generates standardized TREC-style qrels files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import pandas as pd

from fever_ir.data.loaders import ClaimRecord
from fever_ir.data.wiki_parser import ParsedWikiPage, unescape_ptb
from fever_ir.utils.io import write_jsonl


@dataclass(frozen=True)
class QrelsRecord:
    qid: int
    label: str
    gold_doc_ids: list[str]
    evidence_sets: list[list[str]]

    def to_dict(self) -> dict:
        return {
            "qid": self.qid,
            "label": self.label,
            "gold_doc_ids": self.gold_doc_ids,
            "evidence_sets": self.evidence_sets,
        }


def format_doc_id(page_title: str, sent_id: int) -> str:
    """Format canonical document identifier: '{page_title}::{sent_id}'."""
    return f"{page_title}::{sent_id}"


def build_corpus_and_qrels(
    pages: Sequence[ParsedWikiPage],
    claims: Sequence[ClaimRecord],
    use_title_augmentation: bool = True,
) -> tuple[pd.DataFrame, list[QrelsRecord], list[dict]]:
    """Compile corpus rows and map claims to qrels.

    Returns:
        corpus_df: DataFrame with columns doc_idx, doc_id, page_title, sent_id, text, text_with_title
        qrels_records: Valid QrelsRecord list
        dropped_claims: Log of dropped claims and rationale
    """
    rows = []
    seen_doc_ids: set[str] = set()
    doc_idx = 0

    for page in pages:
        display_title = unescape_ptb(page.page_title.replace("_", " "))
        for sentence in page.sentences:
            if not sentence.text.strip():
                continue

            doc_id = format_doc_id(page.page_title, sentence.sent_id)
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)

            text_clean = unescape_ptb(sentence.text)
            text_with_title = f"{display_title}: {text_clean}" if use_title_augmentation else text_clean

            rows.append(
                {
                    "doc_idx": doc_idx,
                    "doc_id": doc_id,
                    "page_title": page.page_title,
                    "sent_id": sentence.sent_id,
                    "text": text_clean,
                    "text_with_title": text_with_title,
                }
            )
            doc_idx += 1

    corpus_df = pd.DataFrame(rows)

    qrels_records: list[QrelsRecord] = []
    dropped_claims: list[dict] = []

    for claim in claims:
        if claim.is_nei:
            continue

        gold_doc_ids = [format_doc_id(page, sent_id) for page, sent_id in claim.flattened_gold_sentences]
        missing = [doc_id for doc_id in gold_doc_ids if doc_id not in seen_doc_ids]

        if missing:
            dropped_claims.append(
                {"claim_id": claim.id, "reason": "missing_evidence_in_corpus", "missing_doc_ids": missing}
            )
            continue

        ev_sets_doc_ids: list[list[str]] = []
        for ev_set in claim.evidence_sets:
            ev_list = [
                format_doc_id(item.page_title, item.sentence_id)
                for item in ev_set
                if item.page_title is not None and item.sentence_id is not None
            ]
            if ev_list:
                ev_sets_doc_ids.append(ev_list)

        qrels_records.append(
            QrelsRecord(
                qid=claim.id,
                label=claim.label,
                gold_doc_ids=sorted(gold_doc_ids),
                evidence_sets=ev_sets_doc_ids,
            )
        )

    return corpus_df, qrels_records, dropped_claims
