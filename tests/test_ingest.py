"""Chunking and ingestion."""

from __future__ import annotations

import io

from ragverify.ingest import Document, build_index, chunk_page, clean_text, load_documents
from ragverify.tokens import count_tokens


class TestCleanText:
    def test_preserves_paragraph_breaks(self):
        # Collapsing ALL whitespace including newlines would destroy the
        # paragraph boundaries the chunker splits on.
        out = clean_text("First para.\n\nSecond para.")
        assert "\n\n" in out

    def test_collapses_runs_of_spaces(self):
        assert clean_text("a     b\t\tc") == "a b c"

    def test_rejoins_hyphenated_line_breaks(self):
        assert "requirements" in clean_text("require-\nments are listed")

    def test_normalizes_crlf(self):
        assert "\r" not in clean_text("a\r\n\r\nb")


class TestChunkPage:
    def test_respects_token_budget(self):
        text = "\n\n".join(f"Paragraph {i} with a handful of words in it." for i in range(60))
        for piece in chunk_page(text, chunk_tokens=100, overlap_tokens=20):
            # Allow the last paragraph to cross slightly; it is admitted whole.
            assert count_tokens(piece) <= 140

    def test_keeps_paragraphs_intact_when_they_fit(self):
        text = "First paragraph here.\n\nSecond paragraph here."
        assert chunk_page(text, chunk_tokens=500, overlap_tokens=50) == [text]

    def test_splits_oversized_single_paragraph(self):
        assert len(chunk_page("word " * 900, chunk_tokens=100, overlap_tokens=10)) > 1

    def test_empty_and_whitespace(self):
        assert chunk_page("", 100, 10) == []
        assert chunk_page("   \n\n  ", 100, 10) == []

    def test_overlap_clamped_below_chunk_size(self):
        # An overlap >= chunk size would otherwise loop forever.
        assert chunk_page("word " * 300, chunk_tokens=50, overlap_tokens=500)


class TestBuildIndex:
    def test_ids_unique_across_same_named_docs(self):
        docs = [
            Document(name="report.txt", pages=["Alpha content about revenue."]),
            Document(name="report.txt", pages=["Beta content about headcount."]),
        ]
        ids = [c.chunk_id for c in build_index(docs, 200, 20)]
        assert len(ids) == len(set(ids)), "colliding ids would misattribute citations"

    def test_page_numbers_carried_for_multipage(self):
        doc = Document(name="a.pdf", pages=["Page one text here.", "Page two text here."])
        chunks = build_index([doc], 200, 20)
        assert {c.page for c in chunks} == {1, 2}
        assert "p.1" in chunks[0].label

    def test_single_page_has_no_page_label(self):
        chunks = build_index([Document(name="a.txt", pages=["Only page."])], 200, 20)
        assert chunks[0].page is None
        assert "#1" in chunks[0].label

    def test_ids_stable_across_rebuilds(self):
        doc = Document(name="a.txt", pages=["Stable content for hashing."])
        assert [c.chunk_id for c in build_index([doc], 200, 20)] == [
            c.chunk_id for c in build_index([doc], 200, 20)
        ]


class TestLoadDocuments:
    def test_reads_plaintext(self):
        f = io.BytesIO(b"Hello world content.")
        f.name = "note.txt"
        docs = load_documents([f])
        assert docs[0].text == "Hello world content."

    def test_falls_back_on_bad_encoding(self):
        f = io.BytesIO(b"caf\xe9 latin-1 bytes")
        f.name = "note.txt"
        assert load_documents([f])[0].text

    def test_one_bad_file_does_not_lose_the_others(self):
        # A PdfReader exception must not propagate out and kill the run.
        good = io.BytesIO(b"Readable content.")
        good.name = "good.txt"
        bad = io.BytesIO(b"not really a pdf")
        bad.name = "broken.pdf"

        docs = load_documents([bad, good])
        assert any("Readable content." in d.text for d in docs)
        assert any("unreadable" in d.name for d in docs)

    def test_empty_input(self):
        assert load_documents([]) == []
