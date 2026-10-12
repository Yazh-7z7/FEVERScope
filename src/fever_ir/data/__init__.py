"""Data acquisition, parsing, sampling, and corpus construction for FEVERScope."""

from fever_ir.data.loaders import load_claims_stream, ClaimRecord
from fever_ir.data.wiki_parser import parse_wiki_line, parse_wiki_page, ParsedWikiSentence
from fever_ir.data.sampling import sample_claims, select_distractor_pages
from fever_ir.data.corpus_builder import build_corpus_and_qrels

__all__ = [
    "load_claims_stream",
    "ClaimRecord",
    "parse_wiki_line",
    "parse_wiki_page",
    "ParsedWikiSentence",
    "sample_claims",
    "select_distractor_pages",
    "build_corpus_and_qrels",
]
