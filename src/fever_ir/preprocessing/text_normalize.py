"""Text normalization and title augmentation utilities.

Follows Single Responsibility Principle (SRP):
Normalizes strings, handles Unicode consistency, and applies title prefixing
for lexical and dense retrieval text streams.
"""

import unicodedata
from fever_ir.data.wiki_parser import unescape_ptb


def normalize_text(text: str, lowercase: bool = True) -> str:
    """Normalize unicode, un-escape PTB brackets, and optionally lowercase."""
    nfc = unicodedata.normalize("NFC", text)
    unescaped = unescape_ptb(nfc)
    return unescaped.lower() if lowercase else unescaped


def make_title_augmented_text(page_title: str, sentence_text: str) -> str:
    """Combine page title and sentence text into '{title with spaces}: {sentence}'."""
    clean_title = unescape_ptb(page_title.replace("_", " "))
    return f"{clean_title}: {sentence_text.strip()}"
