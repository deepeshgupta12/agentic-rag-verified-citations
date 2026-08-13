"""Run configuration.

Writing a user-supplied key into ``os.environ`` is a leak: in a multi-session
Streamlit process one visitor's key becomes the process-wide default for the
next visitor. The key rides in an immutable ``Settings`` object passed
explicitly to the client that needs it, and is never written to the
environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

# Per-million-token pricing used for the cost meter. Unknown models fall back
# to zero so an unlisted model reports "unmetered" instead of a wrong number.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}

DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"

# Public SearxNG instances, tried in order. Several are listed because any
# single public instance is frequently rate-limited, and with no fallback a
# 429 takes down the whole run.
DEFAULT_SEARX_ENDPOINTS = (
    "https://searxng.site/search",
    "https://searx.be/search",
    "https://priv.au/search",
)


@dataclass(frozen=True)
class Settings:
    # --- credentials -----------------------------------------------------
    api_key: str = ""
    base_url: str | None = None

    # --- models ----------------------------------------------------------
    model: str = DEFAULT_MODEL
    embed_model: str = DEFAULT_EMBED_MODEL
    temperature: float = 0.1

    # --- adaptive loop ---------------------------------------------------
    # The whole point of v2: how many escalation rounds the loop may spend
    # before it must answer with whatever it has.
    max_rounds: int = 3
    # Fraction of claims that must survive grounding for a draft to be
    # eligible for "sufficient". Guards against a lenient verifier.
    min_support_rate: float = 0.6

    # --- retrieval -------------------------------------------------------
    chunk_tokens: int = 320
    chunk_overlap_tokens: int = 64
    top_k: int = 6
    # Round 2+ widens retrieval by this factor before escalating source.
    widen_factor: int = 2
    use_embeddings: bool = True
    # Token ceiling for the evidence block in a single prompt. Evidence is
    # packed up to this budget instead of being truncated to 300 characters
    # per chunk.
    evidence_token_budget: int = 6000

    # --- budget ----------------------------------------------------------
    # Hard caps. An escalating loop can otherwise run away: each round costs
    # more than the last, and a corpus that never satisfies the verifier keeps
    # buying rounds until max_rounds. Checked before escalating, so the loop
    # stops at a clean boundary with a usable answer.
    max_seconds: float = 180.0
    max_cost_usd: float = 1.00
    max_calls: int = 40

    # --- safety ----------------------------------------------------------
    # Neutralize model-directed instructions found in retrieved text. Uploaded
    # files and fetched pages are attacker-controllable input.
    sanitize_sources: bool = True
    # Abstain rather than answer when grounding stays below this after the
    # final round. Set to 0.0 to always answer with a confidence label.
    abstain_below_support: float = 0.25

    # --- web -------------------------------------------------------------
    web_enabled: bool = True
    searx_endpoints: tuple[str, ...] = DEFAULT_SEARX_ENDPOINTS
    web_max_results: int = 8
    request_timeout_s: float = 12.0
    max_retries: int = 3

    # --- misc ------------------------------------------------------------
    seed: int | None = 7
    extra: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls, **overrides) -> Settings:
        """Build settings from environment, then apply explicit overrides.

        Overrides that are ``None`` or empty strings are dropped so a blank
        Streamlit text input does not clobber a value set in ``.env``.
        """
        env_endpoints = os.getenv("RAGVERIFY_SEARX_ENDPOINTS", "")
        base = cls(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            model=os.getenv("RAGVERIFY_MODEL", DEFAULT_MODEL),
            embed_model=os.getenv("RAGVERIFY_EMBED_MODEL", DEFAULT_EMBED_MODEL),
            searx_endpoints=(
                tuple(u.strip() for u in env_endpoints.split(",") if u.strip())
                or DEFAULT_SEARX_ENDPOINTS
            ),
        )
        clean = {k: v for k, v in overrides.items() if v not in (None, "")}
        return replace(base, **clean) if clean else base

    def price_per_million(self) -> tuple[float, float] | None:
        """(input, output) price, or None when the model is unpriced."""
        return MODEL_PRICING.get(self.model)

    def supports_temperature(self) -> bool:
        """Reasoning models reject an explicit ``temperature``.

        Sending ``temperature`` to a reasoning model is an API error, so the
        parameter is omitted for those model families.
        """
        return not self.model.startswith(("gpt-5", "o1", "o3", "o4"))
