"""File I/O utilities for streaming and serialized data.

Follows Single Responsibility Principle (SRP):
Provides robust, stream-friendly read/write operations for JSONL and Parquet
without holding entire multi-GB files in memory.
"""

from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any


def read_jsonl_stream(path: Path | str) -> Iterator[dict[str, Any]]:
    """Stream JSON objects from a JSONL file one line at a time.

    Args:
        path: Path to the JSONL file.

    Yields:
        Parsed JSON dictionary for each non-empty line.
    """
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def write_jsonl(records: Iterator[dict[str, Any]] | list[dict[str, Any]], path: Path | str) -> int:
    """Write records to a JSONL file, creating parent directories if needed.

    Args:
        records: Iterable of dictionaries to write.
        path: Target file path.

    Returns:
        Total number of records written.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with file_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
