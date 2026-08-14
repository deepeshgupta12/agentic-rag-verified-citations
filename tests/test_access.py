"""Tenant isolation, document ACLs, and the audit trail.

Everything else in this project protects the reader from a wrong answer.
This protects them from a *correct* answer drawn from a document they were
never allowed to see — the failure a verification pipeline makes worse, since
the answer arrives grounded, cited and confidently supported by a source the
reader has no right to.
"""

from __future__ import annotations

import json

from ragverify.access import AccessPolicy, DocumentACL, OpenPolicy, Principal
from ragverify.audit import AuditEvent, AuditLog
from ragverify.config import Settings
from ragverify.ingest import Document
from ragverify.orchestrator import Corpus

STAFF = Principal(subject="alice", tenant="acme", roles=frozenset({"staff"}))
EXEC = Principal(subject="bob", tenant="acme", roles=frozenset({"exec"}))
OUTSIDER = Principal(subject="eve", tenant="other", roles=frozenset({"exec", "admin"}))

DOCS = [
    Document(name="public.txt", pages=["European revenue grew 34% year over year."]),
    Document(name="secret.txt", pages=["The acquisition of Northwind closes in March."]),
    Document(name="unclassified.txt", pages=["An internal draft with no ACL assigned."]),
]

POLICY = AccessPolicy([
    DocumentACL(doc_name="public.txt", tenant="acme", tenant_readable=True),
    DocumentACL(doc_name="secret.txt", tenant="acme", allow_roles=frozenset({"exec"})),
])


def visible(principal) -> set[str]:
    corpus = Corpus(DOCS, Settings(use_embeddings=False), principal=principal, policy=POLICY)
    return {c.doc_name for c in corpus.chunks}


class TestAccessControl:
    def test_role_gates_a_restricted_document(self):
        assert visible(STAFF) == {"public.txt"}
        assert visible(EXEC) == {"public.txt", "secret.txt"}

    def test_unclassified_documents_are_denied_to_everyone(self):
        """Missing ACLs are usually accidents; treating them as public makes
        every ingestion mistake a disclosure."""
        assert "unclassified.txt" not in visible(EXEC)
        assert "unclassified.txt" not in visible(STAFF)

    def test_tenant_isolation_is_not_overridable_by_role(self):
        """A global admin role must not cross a tenant boundary, or the
        boundary is decorative."""
        assert visible(OUTSIDER) == set()

    def test_subject_grant(self):
        policy = AccessPolicy([
            DocumentACL(doc_name="secret.txt", tenant="acme",
                        allow_subjects=frozenset({"alice"})),
        ])
        assert policy.permits(STAFF, "secret.txt")
        assert not policy.permits(EXEC, "secret.txt")

    def test_filtering_happens_before_scoring(self):
        """Post-filtering a ranked list leaks through the ranking itself."""
        corpus = Corpus(DOCS, Settings(use_embeddings=False), principal=STAFF, policy=POLICY)
        hits = corpus.retriever.search("Northwind acquisition closes", top_k=5)
        assert all(h.chunk.doc_name == "public.txt" for h in hits)

    def test_denials_are_reported_to_the_caller(self):
        corpus = Corpus(DOCS, Settings(use_embeddings=False), principal=STAFF, policy=POLICY)
        assert "secret.txt" in corpus.denied_documents
        assert any("withheld" in w for w in corpus.warnings)

    def test_no_policy_leaves_behaviour_unchanged(self):
        """Access control must be opt-in and inert when unused."""
        corpus = Corpus(DOCS, Settings(use_embeddings=False))
        assert {c.doc_name for c in corpus.chunks} == {
            "public.txt", "secret.txt", "unclassified.txt"
        }
        assert corpus.denied_documents == []

    def test_open_policy_is_explicit(self):
        """Running without access control should be visible in the code."""
        corpus = Corpus(DOCS, Settings(use_embeddings=False),
                        principal=OUTSIDER, policy=OpenPolicy())
        assert len({c.doc_name for c in corpus.chunks}) == 3


class TestAuditLog:
    def test_records_are_chained(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        log.write(AuditEvent.QUERY, "alice", "acme", question_hash="abc")
        log.write(AuditEvent.ANSWERED, "alice", "acme", outcome="answered")

        assert log.records[1].prev_hash == log.records[0].record_hash
        assert AuditLog.verify(tmp_path / "audit.jsonl")["valid"]

    def test_tampering_is_detected_and_located(self, tmp_path):
        """A verifier that says only 'invalid' is unusable when it matters:
        someone needs to know which records are still trustworthy."""
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        for i in range(4):
            log.write(AuditEvent.QUERY, "alice", "acme", question_hash=f"q{i}")

        lines = path.read_text().splitlines()
        record = json.loads(lines[2])
        record["detail"]["question_hash"] = "tampered"
        lines[2] = json.dumps(record)
        path.write_text("\n".join(lines) + "\n")

        report = AuditLog.verify(path)
        assert not report["valid"]
        assert report["broken_at"] == 3

    def test_deleted_record_breaks_the_chain(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        for i in range(4):
            log.write(AuditEvent.QUERY, "alice", "acme", question_hash=f"q{i}")

        lines = path.read_text().splitlines()
        del lines[1]
        path.write_text("\n".join(lines) + "\n")

        assert not AuditLog.verify(path)["valid"]

    def test_chain_continues_across_processes(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        AuditLog(path).write(AuditEvent.QUERY, "alice", "acme")
        AuditLog(path).write(AuditEvent.ANSWERED, "alice", "acme")

        assert AuditLog.verify(path)["valid"]
        assert AuditLog.verify(path)["records"] == 2

    def test_reads_are_tenant_scoped(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        log.write(AuditEvent.QUERY, "alice", "acme")
        log.write(AuditEvent.QUERY, "eve", "other")

        assert len(AuditLog.read(path, tenant="acme")) == 1
        assert len(AuditLog.read(path)) == 2

    def test_disabled_log_writes_nothing(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path, enabled=False)
        log.write(AuditEvent.QUERY, "alice", "acme")
        assert not path.exists()

    def test_question_is_hashed_not_stored(self, tmp_path):
        """A question is often sensitive in itself; storing it verbatim makes
        the audit log a second copy of what the ACLs were protecting."""
        import sys

        sys.path.insert(0, "tests")
        from test_orchestrator import (
            CITED_ANSWER, GOOD_DRAFT, QUESTION, FakeLLM,
            corpus_for, settings, triage, verdict,
        )

        from ragverify.audit import record_run
        from ragverify.orchestrator import AdaptiveResearcher
        from ragverify.schemas import NextAction, Verdict

        cfg = settings(max_rounds=1)
        llm = FakeLLM(cfg, [
            triage(), GOOD_DRAFT, verdict(Verdict.SUFFICIENT, NextAction.ANSWER), CITED_ANSWER,
        ])
        result = AdaptiveResearcher(cfg, llm, corpus_for(cfg)).run(QUESTION)

        path = tmp_path / "audit.jsonl"
        record_run(AuditLog(path), STAFF, QUESTION, result, granted=["report.txt"])

        raw = path.read_text()
        assert QUESTION not in raw, "the question must not be stored verbatim"
        assert "question_hash" in raw
        assert AuditLog.verify(path)["valid"]
