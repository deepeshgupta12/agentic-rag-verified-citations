"""Document loading and token-aware chunking.

Two decisions here are load-bearing.

1. Chunks were 800 *words* (roughly 1,100 tokens) but the prompt builder only
   ever emitted ``chunk.text[:300]`` -- the first 300 *characters*. Around 97%
   of every retrieved chunk was silently discarded before the model saw it, so
   retrieval could be perfect and the answer would still be built on the first
   two sentences of each hit. Chunks here are sized in tokens, and the prompt
   builder packs whole chunks against a token budget.

2. PDF page numbers were dropped, so a citation could never point anywhere
   more precise than the filename. Page is carried through to the chunk.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from . import tables as tables_mod
from .schemas import Chunk
from .tokens import count_tokens, split_tokens

_WS = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n{3,}")
# A page's worth of ligatures and hyphenation artefacts that PDF extraction
# leaves behind and that wreck tokenisation for the lexical retriever.
_DEHYPHEN = re.compile(r"(\w)-\n(\w)")


class _Readable(Protocol):
    name: str

    def read(self) -> bytes: ...


@dataclass
class Document:
    name: str
    pages: list[str]

    @property
    def text(self) -> str:
        return "\n\n".join(self.pages)

    @property
    def n_pages(self) -> int:
        return len(self.pages)


def clean_text(text: str) -> str:
    """Normalise whitespace without destroying paragraph structure.

    Collapsing *all* whitespace including newlines would erase paragraph
    boundaries -- the main signal available for splitting on a semantic
    boundary rather than mid-sentence.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _DEHYPHEN.sub(r"\1\2", text)
    text = _WS.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def _doc_id(name: str, text: str) -> str:
    digest = hashlib.sha1(f"{name}:{text[:4096]}".encode()).hexdigest()
    return digest[:8]


def load_documents(files: Iterable[_Readable]) -> list[Document]:
    """Read uploaded files into per-page text.

    Never raises for a single bad file: a corrupt or encrypted PDF among ten
    good ones should cost you that one document, not the run. Letting the
    ``PdfReader`` exception propagate would kill the whole request.
    """
    documents: list[Document] = []
    for file in files:
        name = getattr(file, "name", "document")
        try:
            if name.lower().endswith(".pdf"):
                pages = _read_pdf(file)
            else:
                pages = _read_plaintext(file)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a warning
            documents.append(Document(name=f"{name} (unreadable: {exc})", pages=[]))
            continue

        pages = [clean_text(p) for p in pages]
        if any(p for p in pages):
            documents.append(Document(name=name, pages=pages))
    return documents


def _read_pdf(file) -> list[str]:
    from pypdf import PdfReader  # imported lazily so core stays importable

    reader = PdfReader(file)
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001
            raise ValueError("encrypted PDF") from exc
    return [(page.extract_text() or "") for page in reader.pages]


def _read_plaintext(file) -> list[str]:
    raw = file.read()
    if isinstance(raw, str):
        return [raw]
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return [raw.decode(encoding)]
        except UnicodeDecodeError:
            continue
    return [raw.decode("utf-8", errors="replace")]


def chunk_page(
    text: str,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Split on paragraph boundaries, packing up to ``chunk_tokens``.

    Paragraphs are kept whole where they fit so a chunk rarely begins
    mid-sentence; only a single paragraph larger than the budget is hard-split.
    """
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    overlap_tokens = max(0, min(overlap_tokens, chunk_tokens - 1))

    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    def flush() -> None:
        nonlocal buffer, buffer_tokens
        if buffer:
            chunks.append("\n\n".join(buffer).strip())
            buffer, buffer_tokens = [], 0

    for para in paragraphs:
        n = count_tokens(para)
        if n > chunk_tokens:
            flush()
            chunks.extend(split_tokens(para, chunk_tokens, overlap_tokens))
            continue
        if buffer_tokens + n > chunk_tokens:
            # Carry the tail of the current buffer into the next chunk so a
            # fact spanning the boundary is still retrievable from one chunk.
            tail = buffer[-1] if buffer and overlap_tokens else None
            flush()
            if tail and count_tokens(tail) <= overlap_tokens:
                buffer, buffer_tokens = [tail], count_tokens(tail)
        buffer.append(para)
        buffer_tokens += n

    flush()
    return [c for c in chunks if c]


def build_index(
    documents: Sequence[Document],
    chunk_tokens: int = 320,
    overlap_tokens: int = 64,
    table_aware: bool = True,
) -> list[Chunk]:
    """Flatten documents into globally-addressable chunks.

    ``chunk_id`` embeds a document hash so ids stay stable across runs and
    unique across documents that share a filename -- citations resolve by id,
    so a collision would attribute a claim to the wrong file.
    """
    index: list[Chunk] = []
    for doc in documents:
        did = _doc_id(doc.name, doc.text)
        ordinal = 0
        for page_no, page_text in enumerate(doc.pages, start=1):
            # Table rows are chunked individually and carry their headers, so
            # a row retrieved alone still says what its numbers measure.
            # Paragraph chunking splits a table wherever the token budget
            # lands, which can strand "Eastern 250 9% 190" several chunks from
            # the header row: retrievable, groundable, and meaningless.
            table_lines: set[int] = set()
            if table_aware:
                blocks = tables_mod.find_tables(page_text)
                table_lines = tables_mod.table_line_span(blocks)
                for block in blocks:
                    for row in block.rows:
                        ordinal += 1
                        text = row.as_text()
                        if block.caption:
                            text = f"{block.caption} — {text}"
                        index.append(Chunk(
                            chunk_id=f"{did}-{ordinal}", doc_name=doc.name,
                            ordinal=ordinal, text=text, n_tokens=count_tokens(text),
                            page=page_no if doc.n_pages > 1 else None,
                            table_id=row.table_id, row_index=row.row_index,
                            headers=row.headers, table_caption=block.caption,
                        ))

            prose = page_text
            if table_lines:
                prose = "\n".join(
                    line for n, line in enumerate(page_text.splitlines())
                    if n not in table_lines
                )

            for piece in chunk_page(prose, chunk_tokens, overlap_tokens):
                ordinal += 1
                index.append(
                    Chunk(
                        chunk_id=f"{did}-{ordinal}",
                        doc_name=doc.name,
                        ordinal=ordinal,
                        text=piece,
                        n_tokens=count_tokens(piece),
                        page=page_no if doc.n_pages > 1 else None,
                    )
                )
    return index


def corpus_summary(index: Sequence[Chunk]) -> str:
    if not index:
        return "No local documents provided."
    names = sorted({c.doc_name for c in index})
    tokens = sum(c.n_tokens for c in index)
    listed = ", ".join(names[:8]) + (f" (+{len(names) - 8} more)" if len(names) > 8 else "")
    return f"{len(names)} document(s): {listed} | {len(index)} chunks, ~{tokens:,} tokens"


def top_terms(index: Sequence[Chunk], limit: int = 40) -> list[str]:
    """Most distinctive corpus terms, shown to triage so its route decision
    is informed by what the documents are actually about.

    Handing triage only filenames and a chunk count would leave it guessing
    at coverage from a string like ``q3-report.pdf``.
    """
    from collections import Counter

    from .retrieval import STOPWORDS, tokenize

    counts: Counter[str] = Counter()
    for chunk in index:
        counts.update(t for t in set(tokenize(chunk.text)) if t not in STOPWORDS and len(t) > 3)
    return [term for term, _ in counts.most_common(limit)]
