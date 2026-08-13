"""RagVerify — agentic RAG with verified citations.

A research pipeline that will not answer until its citations survive a
mechanical grounding check, escalating its retrieval strategy until the
evidence holds or the round budget runs out.
"""

from .config import Settings
from .ingest import Document, build_index, load_documents
from .orchestrator import AdaptiveResearcher, Corpus, research
from .schemas import (
    Claim,
    EvidenceItem,
    GroundingReport,
    ResearchResult,
    Route,
    Verdict,
)
from .trace import Event, EventKind, Tracer

__version__ = "1.0.0"

__all__ = [
    "AdaptiveResearcher",
    "Claim",
    "Corpus",
    "Document",
    "Event",
    "EventKind",
    "EvidenceItem",
    "GroundingReport",
    "ResearchResult",
    "Route",
    "Settings",
    "Tracer",
    "Verdict",
    "build_index",
    "load_documents",
    "research",
    "__version__",
]
