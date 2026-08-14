"""Tenant isolation and document-level access control.

Everything else in this project protects the reader from a *wrong* answer.
This protects them from a *correct* answer drawn from a document they were
never allowed to see -- which is the failure mode a verification pipeline
makes worse rather than better, because the answer arrives grounded, cited
and confidently supported by a source the reader has no right to.

Retrieval is the leak. A search over a shared index returns whatever matches,
and every downstream stage then faithfully preserves it: chunking, grounding,
the citation label, the evidence ledger. By the time an answer exists the
document is quoted verbatim with a source id pointing at it. There is no
later point at which the leak can be undone.

So the filter belongs at the index, before scoring. Two properties follow:

* **Deny by default.** An unlabelled document is not public; it is
  unclassified, and treating those as readable means every ingestion mistake
  becomes a disclosure. A document with no ACL is visible to nobody until
  someone says otherwise.
* **Filter before retrieval, not after.** Post-filtering a ranked list leaks
  through the ranking itself -- result counts, scores and relevance orderings
  all carry information about documents the caller cannot read. Denied
  documents never enter scoring.

Deliberately not an authentication system. This decides what a *given*
principal may read; establishing who the principal is belongs to the
application, and pretending otherwise would invite someone to trust a
`Principal` object assembled from an unvalidated request.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .schemas import Chunk


@dataclass(frozen=True)
class Principal:
    """Who is asking. Constructed by the application after authentication.

    ``ragverify`` never establishes identity -- it has no session, token or
    user store, and inventing one would encourage callers to trust a
    principal built straight from a request body.
    """

    subject: str
    tenant: str
    roles: frozenset[str] = field(default_factory=frozenset)

    def has_role(self, *roles: str) -> bool:
        return bool(self.roles & set(roles))


@dataclass(frozen=True)
class DocumentACL:
    """Who may read one document.

    Empty ``allow_roles`` and ``allow_subjects`` means nobody but the tenant's
    administrators -- not everybody. The default has to be the safe one,
    because ACLs are most often missing by accident.
    """

    doc_name: str
    tenant: str
    allow_roles: frozenset[str] = field(default_factory=frozenset)
    allow_subjects: frozenset[str] = field(default_factory=frozenset)
    # Marks a document as readable by any authenticated member of its tenant.
    # Requires saying so explicitly; there is no way to reach it by omission.
    tenant_readable: bool = False

    def permits(self, principal: Principal) -> bool:
        # Tenant isolation is checked first and is not overridable by role.
        # A global "admin" role must not read across tenant boundaries, or
        # the boundary is decorative.
        if principal.tenant != self.tenant:
            return False
        if principal.subject in self.allow_subjects:
            return True
        if principal.roles & self.allow_roles:
            return True
        return self.tenant_readable


class AccessPolicy:
    """Resolves whether a principal may read a document.

    Unknown documents are denied. A missing ACL means the document was never
    classified, and reading unclassified material is exactly the disclosure
    this exists to prevent.
    """

    def __init__(self, acls: Iterable[DocumentACL] = ()) -> None:
        self._acls: dict[str, DocumentACL] = {a.doc_name: a for a in acls}
        # Denials are recorded so the caller can tell "no results because
        # nothing matched" from "no results because you may not see them",
        # without the answer itself revealing which.
        self.denied: dict[str, int] = {}

    def add(self, acl: DocumentACL) -> None:
        self._acls[acl.doc_name] = acl

    def acl_for(self, doc_name: str) -> DocumentACL | None:
        return self._acls.get(doc_name)

    def permits(self, principal: Principal, doc_name: str) -> bool:
        acl = self._acls.get(doc_name)
        if acl is None:
            return False  # unclassified is not public
        return acl.permits(principal)

    def filter_chunks(
        self, principal: Principal, chunks: Sequence[Chunk]
    ) -> list[Chunk]:
        """Drop chunks the principal may not read, before any scoring.

        Called at index construction rather than on results: a post-filtered
        ranking still leaks through its own shape, since result counts and
        scores are computed over documents the caller cannot see.
        """
        allowed: list[Chunk] = []
        denied: dict[str, int] = {}
        for chunk in chunks:
            if self.permits(principal, chunk.doc_name):
                allowed.append(chunk)
            else:
                denied[chunk.doc_name] = denied.get(chunk.doc_name, 0) + 1
        self.denied = denied
        return allowed

    def summary(self) -> str:
        """What was withheld, for the caller's logs -- never for the answer."""
        if not self.denied:
            return ""
        total = sum(self.denied.values())
        return (
            f"{total} chunk(s) across {len(self.denied)} document(s) withheld "
            "by access policy."
        )


#: A policy that permits everything, for single-tenant and local use.
#:
#: Named rather than implicit: a caller reaching for this is choosing to run
#: without access control, and that choice should be visible in their code.
class OpenPolicy(AccessPolicy):
    """No access control. Explicit, so it cannot be selected by accident."""

    def permits(self, principal: Principal, doc_name: str) -> bool:  # noqa: ARG002
        return True

    def filter_chunks(self, principal: Principal, chunks: Sequence[Chunk]) -> list[Chunk]:  # noqa: ARG002
        self.denied = {}
        return list(chunks)
