"""Table detection and row-level provenance.

Two problems with treating a table as prose. The citation becomes
uncheckable — "p.3" when the fact is one cell of a forty-row table — and the
row loses its headers, because paragraph chunking splits wherever the token
budget lands. A stranded "Eastern 250 9% 190" is retrievable, grounds fine
because the digits are present, and says nothing about what 250 measures.
"""

from __future__ import annotations

from ragverify.ingest import Document, build_index
from ragverify.tables import find_tables, table_line_span

MARKDOWN = """Acme Segment Results - Q3 2024
All figures in millions of euro unless stated otherwise.

Segment | Revenue | Margin | Headcount
--- | --- | --- | ---
Northern | 1,240 | 21% | 880
Southern | 610 | 14% | 410
Eastern | 250 | 9% | 190

Northern segment margin improved on renewals across the reporting period."""


class TestDetection:
    def test_finds_a_markdown_table(self):
        blocks = find_tables(MARKDOWN)
        assert len(blocks) == 1
        assert blocks[0].headers == ["Segment", "Revenue", "Margin", "Headcount"]
        assert len(blocks[0].rows) == 3

    def test_header_row_is_recognised_by_absence_of_digits(self):
        """A header names columns; a data row measures them."""
        blocks = find_tables(MARKDOWN)
        assert "Northern" not in blocks[0].headers

    def test_caption_is_taken_from_above(self):
        assert "millions of euro" in find_tables(MARKDOWN)[0].caption

    def test_fixed_width_table(self):
        text = (
            "Region        Revenue     Margin\n"
            "Northern      1,240       21%\n"
            "Southern      610         14%\n"
            "Eastern       250         9%"
        )
        blocks = find_tables(text)
        assert blocks and len(blocks[0].rows) >= 3


class TestPrecision:
    """A false table fragments prose and wrecks retrieval for that passage."""

    def test_prose_with_a_stray_pipe_is_not_a_table(self):
        assert not find_tables(
            "Revenue grew strongly this year.\n"
            "The board approved the plan | as expected.\n"
            "Margins held steady across the business."
        )

    def test_plain_paragraphs_are_not_a_table(self):
        assert not find_tables("One two three.\nFour five six.\nSeven eight nine.")

    def test_two_rows_is_not_enough(self):
        assert not find_tables("A | B | C\nD | E | F")

    def test_inconsistent_column_counts_rejected(self):
        assert not find_tables(
            "one | two\nthree | four | five | six | seven\neight\nnine | ten | eleven"
        )

    def test_empty_input(self):
        assert find_tables("") == []
        assert table_line_span([]) == set()


class TestRowText:
    def test_row_carries_its_headers(self):
        """A bare '250' says nothing; 'Revenue: 250' is self-describing."""
        row = find_tables(MARKDOWN)[0].rows[2]
        text = row.as_text()
        assert "Revenue: 250" in text
        assert "Segment: Eastern" in text

    def test_row_label_names_table_and_row(self):
        row = find_tables(MARKDOWN)[0].rows[0]
        assert row.label == "table 1 row 1"


class TestChunking:
    def test_rows_become_individual_chunks(self):
        chunks = build_index([Document(name="q3.txt", pages=[MARKDOWN])])
        rows = [c for c in chunks if c.is_table_row]

        assert len(rows) == 3
        assert all(c.headers for c in rows)

    def test_citation_label_locates_the_row(self):
        chunks = build_index([Document(name="q3.txt", pages=[MARKDOWN])])
        row = next(c for c in chunks if c.is_table_row)
        assert "table 1 row 1" in row.label

    def test_prose_still_chunked_alongside(self):
        chunks = build_index([Document(name="q3.txt", pages=[MARKDOWN])])
        prose = [c for c in chunks if not c.is_table_row]
        assert any("renewals" in c.text for c in prose)

    def test_table_lines_are_not_duplicated_into_prose(self):
        chunks = build_index([Document(name="q3.txt", pages=[MARKDOWN])])
        prose = " ".join(c.text for c in chunks if not c.is_table_row)
        assert "1,240" not in prose, "a row must not appear twice in the index"

    def test_prose_only_documents_are_unchanged(self):
        """Table awareness must be inert when there is no table."""
        doc = Document(name="p.txt", pages=[
            "European revenue grew 34% year over year.\n\n"
            "The Berlin engineering office reached 412 staff at quarter end."
        ])
        assert [c.text for c in build_index([doc], table_aware=True)] == [
            c.text for c in build_index([doc], table_aware=False)
        ]

    def test_row_is_retrievable_by_its_header(self):
        """The point of carrying headers: the row answers a column question."""
        from ragverify.retrieval import HybridRetriever

        chunks = build_index([Document(name="q3.txt", pages=[MARKDOWN])])
        hits = HybridRetriever(chunks).search("Eastern segment revenue", top_k=2)
        assert any("Eastern" in h.chunk.text for h in hits)
