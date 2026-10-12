"""Acceptance and unit tests for FEVER data processing.

Corresponds to Phase 1 Step 6 requirements:
1. doc_id unique; doc_idx is exactly 0..N-1.
2. For every retained claim, every gold (page, sent_id) exists in corpus with non-empty text.
3. No NEI claim appears in qrels.
4. Two runs with the same seed produce byte-identical outputs.
5. Parser reproduces exact sentence text.
6. Title matching works with Unicode NFC normalization.
"""

import pytest
from fever_ir.data.wiki_parser import (
    parse_wiki_line,
    parse_wiki_page,
    normalize_nfc,
    unescape_ptb,
)
from fever_ir.data.corpus_builder import format_doc_id, build_corpus_and_qrels
from fever_ir.data.loaders import ClaimRecord, EvidenceItem


def test_parse_wiki_line_preserves_sent_id() -> None:
    """Verify sent_id is parsed from line content, not line position (§3.2)."""
    raw_line = "3\tThis is the third sentence.\thttps://wiki.org\tThird"
    parsed = parse_wiki_line(raw_line)
    assert parsed is not None
    assert parsed.sent_id == 3
    assert parsed.text == "This is the third sentence."


def test_unescape_ptb_brackets() -> None:
    """Verify PTB escapes like -LRB- and -RRB- are un-escaped correctly."""
    escaped = "Barack Obama -LRB- born August 4 -RRB- was president -COLON- yes"
    expected = "Barack Obama ( born August 4 ) was president : yes"
    assert unescape_ptb(escaped) == expected


def test_nfc_normalization_equality() -> None:
    """Verify Unicode NFC normalization aligns accented characters (§3.2, §5 Step 6)."""
    # Decomposed NFD vs composed NFC
    nfd_title = "Caf\u0065\u0301"  # Cafe + combining acute
    nfc_title = "Caf\u00e9"          # Café single code point
    assert nfd_title != nfc_title
    assert normalize_nfc(nfd_title) == normalize_nfc(nfc_title)


def test_format_doc_id() -> None:
    """Verify canonical doc_id format: '{page_title}::{sent_id}'."""
    assert format_doc_id("Roman_Empire", 4) == "Roman_Empire::4"


def test_no_nei_in_qrels() -> None:
    """Verify NEI claims are never placed into qrels (§2 D5, §5 Step 6)."""
    nei_claim = ClaimRecord(
        id=999,
        verifiable="NOT VERIFIABLE",
        label="NOT ENOUGH INFO",
        claim="An unverifiable statement.",
        evidence_sets=[],
    )
    corpus_df, qrels, dropped = build_corpus_and_qrels(pages=[], claims=[nei_claim])
    assert len(qrels) == 0
    assert 999 not in [q.qid for q in qrels]
