"""Wikipedia page and sentence line parser for FEVER dumps.

Follows Single Responsibility Principle (SRP):
Responsible exclusively for parsing Wikipedia lines, un-escaping PTB markup,
and normalizing Unicode NFC representations according to FEVER specs (§3.2).
"""

from dataclasses import dataclass
import unicodedata

# Standard Penn Treebank escape sequence mapping
PTB_ESCAPES = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]",
    "-LCB-": "{",
    "-RCB-": "}",
    "-COLON-": ":",
}


@dataclass(frozen=True)
class ParsedWikiSentence:
    sent_id: int
    text: str


@dataclass(frozen=True)
class ParsedWikiPage:
    page_title: str
    sentences: list[ParsedWikiSentence]


def normalize_nfc(text: str) -> str:
    """Normalize text using Unicode Normalization Form C (NFC)."""
    return unicodedata.normalize("NFC", text)


def unescape_ptb(text: str) -> str:
    """Replace Penn Treebank escape sequences with standard characters."""
    result = text
    for escape_seq, char in PTB_ESCAPES.items():
        result = result.replace(escape_seq, char)
    return result


def parse_wiki_line(line: str) -> ParsedWikiSentence | None:
    """Parse a single line from Wikipedia lines string.

    Format expected: sent_id <TAB> sentence_text <TAB> hyperlinks...
    sent_id must be parsed from the line itself, never inferred from index (§3.2).
    """
    stripped = line.strip()
    if not stripped:
        return None

    parts = stripped.split("\t")
    if len(parts) < 2:
        return None

    try:
        sent_id = int(parts[0])
    except ValueError:
        return None

    sentence_text = normalize_nfc(parts[1])
    return ParsedWikiSentence(sent_id=sent_id, text=sentence_text)


def parse_wiki_page(raw_page: dict[str, str]) -> ParsedWikiPage:
    """Parse raw Wikipedia JSON record into ParsedWikiPage."""
    raw_title = raw_page.get("id", "")
    page_title = normalize_nfc(raw_title)

    lines_str = raw_page.get("lines", "")
    sentences: list[ParsedWikiSentence] = []

    for line in lines_str.split("\n"):
        parsed = parse_wiki_line(line)
        if parsed is not None:
            sentences.append(parsed)

    return ParsedWikiPage(page_title=page_title, sentences=sentences)
