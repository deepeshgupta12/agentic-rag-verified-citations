"""Table detection and row-level provenance.

A citation resolves to a chunk, and a chunk is a span of prose. That works
until the fact lives in a table, where it produces two distinct problems.

**The citation is imprecise.** "acme_q3.pdf p.3" tells a reader which page to
search when the answer is one cell of a forty-row table. The provenance
exists but is not usable for checking, which is most of what provenance is
for.

**The row loses its headers.** Paragraph chunking splits a table wherever the
token budget lands, so a chunk can contain "Eastern 250 9% 190" with the
header row several chunks away. The number is retrievable and meaningless:
nothing in that span says 250 is revenue in millions rather than headcount.
Worse, it grounds fine -- the digits are present -- so a claim citing it
passes every check while resting on a column the pipeline cannot identify.

Rows are therefore chunked individually and carry their headers inline, so a
row is self-describing wherever it is retrieved, and its citation names the
table and row rather than the page.

Detection is conservative. A false table turns ordinary prose into fragments
and wrecks retrieval, which is far more damaging than missing a real one, so
a block must look unambiguously tabular before it is treated as such.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# A cell separator: the pipe form produced by HTML extraction and markdown,
# or three-plus spaces from fixed-width plain-text tables.
_PIPE = re.compile(r"\s*\|\s*")
_WIDE_GAP = re.compile(r"\S {3,}\S")

# Markdown rule rows: |---|---|. Marks the header boundary explicitly.
_RULE = re.compile(r"^[\s|:+-]*[-+][\s|:+-]*$")

# A table needs this many rows before it is one. Two lines sharing a pipe are
# more often prose containing a pipe than a table.
MIN_ROWS = 3
MIN_COLUMNS = 2


@dataclass
class TableRow:
    table_id: int
    row_index: int
    cells: list[str]
    headers: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        """The row rendered so it is self-describing in isolation.

        Header-labelled where headers are known, because a bare "Eastern 250
        9% 190" retrieved on its own says nothing about what 250 measures.
        """
        if self.headers and len(self.headers) == len(self.cells):
            pairs = [f"{h}: {c}" for h, c in zip(self.headers, self.cells, strict=True) if c]
            return " | ".join(pairs)
        return " | ".join(c for c in self.cells if c)

    @property
    def label(self) -> str:
        return f"table {self.table_id + 1} row {self.row_index + 1}"


@dataclass
class TableBlock:
    table_id: int
    headers: list[str]
    rows: list[TableRow]
    start_line: int
    end_line: int
    caption: str = ""


def _split_cells(line: str) -> list[str]:
    if "|" in line:
        cells = [c.strip() for c in _PIPE.split(line.strip().strip("|"))]
    else:
        cells = [c.strip() for c in re.split(r"\s{3,}", line.strip())]
    return [c for c in cells if c or len(cells) > 1]


def _looks_tabular(line: str) -> bool:
    stripped = line.strip()
    if not stripped or _RULE.match(stripped):
        return False
    if "|" in stripped:
        return len(_split_cells(stripped)) >= MIN_COLUMNS
    return bool(_WIDE_GAP.search(stripped)) and len(_split_cells(stripped)) >= MIN_COLUMNS


def find_tables(text: str) -> list[TableBlock]:
    """Locate table blocks. Conservative by design.

    A false positive fragments prose into cells and destroys retrieval for
    that passage, which costs far more than missing a table would.
    """
    lines = text.splitlines()
    blocks: list[TableBlock] = []
    run: list[tuple[int, str]] = []

    def flush() -> None:
        if len(run) < MIN_ROWS:
            run.clear()
            return

        rows_raw = [(i, _split_cells(text_line)) for i, text_line in run]
        width = max(len(cells) for _, cells in rows_raw)
        if width < MIN_COLUMNS:
            run.clear()
            return

        # Consistency check: a genuine table has a stable column count. Prose
        # that happens to contain separators does not.
        consistent = sum(1 for _, cells in rows_raw if len(cells) == width)
        if consistent / len(rows_raw) < 0.6:
            run.clear()
            return

        table_id = len(blocks)
        first_index, first_cells = rows_raw[0]
        # The first row is a header when it carries no digits: a header names
        # columns, a data row measures them.
        header_like = all(not re.search(r"\d", c) for c in first_cells if c)
        headers = first_cells if header_like and len(first_cells) == width else []
        body = rows_raw[1:] if headers else rows_raw

        rows = [
            TableRow(table_id=table_id, row_index=n, cells=cells, headers=headers)
            for n, (_, cells) in enumerate(body)
        ]
        blocks.append(TableBlock(
            table_id=table_id, headers=headers, rows=rows,
            start_line=first_index, end_line=run[-1][0],
        ))
        run.clear()

    for index, line in enumerate(lines):
        if _looks_tabular(line):
            run.append((index, line))
        elif _RULE.match(line.strip()):
            continue  # a rule row belongs to the table it separates
        else:
            flush()
    flush()

    # Attach a caption from the line above, where one reads like a title.
    for block in blocks:
        for offset in (1, 2):
            candidate = block.start_line - offset
            if 0 <= candidate < len(lines):
                text_line = lines[candidate].strip()
                if text_line and not _looks_tabular(text_line) and len(text_line) < 120:
                    block.caption = text_line
                    break

    return blocks


def table_line_span(blocks: list[TableBlock]) -> set[int]:
    """Line numbers belonging to a table, so prose chunking can skip them."""
    covered: set[int] = set()
    for block in blocks:
        covered.update(range(block.start_line, block.end_line + 1))
    return covered
