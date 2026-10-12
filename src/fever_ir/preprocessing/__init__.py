"""Text normalization, preprocessing, and tokenization modules."""

from fever_ir.preprocessing.text_normalize import normalize_text, make_title_augmented_text
from fever_ir.preprocessing.tokenize import tokenize_text

__all__ = ["normalize_text", "make_title_augmented_text", "tokenize_text"]
