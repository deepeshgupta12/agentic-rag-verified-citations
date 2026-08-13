"""Optional AG2 group-chat backend.

Calling five agents with one isolated ``generate_reply`` each is not agent
teamwork: no ``GroupChat``, no handoffs, no shared history, so no agent ever
reads another's output. That pattern gains nothing a direct API call would not.

``orchestrator.py`` uses direct calls on purpose -- it needs to branch on typed
results, which free-form chat does not give you. This module
exists for the part that genuinely benefits from multi-agent conversation:
adversarial review, where a critic and a defender argue over a draft and the
disagreement itself is the signal.

Import is lazy and entirely optional; nothing in the core path requires AG2.
"""

from __future__ import annotations

from typing import Any

from .config import Settings
from .schemas import EvidenceItem, ResearchDraft


def ag2_available() -> bool:
    try:
        import autogen  # noqa: F401

        return True
    except ImportError:
        return False


def build_llm_config(settings: Settings) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "api_type": "openai",
        "model": settings.model,
        "api_key": settings.api_key,
    }
    if settings.base_url:
        entry["base_url"] = settings.base_url

    config: dict[str, Any] = {"config_list": [entry], "cache_seed": settings.seed}
    # Reasoning models reject an explicit temperature, so it is gated.
    if settings.supports_temperature():
        config["temperature"] = settings.temperature
    return config


def adversarial_review(
    question: str,
    draft: ResearchDraft,
    evidence: list[EvidenceItem],
    settings: Settings,
    max_turns: int = 4,
) -> str | None:
    """Run a critic/defender debate over a draft.

    Returns the transcript, or ``None`` when AG2 is not installed. This is a
    genuine use of a group chat: the two agents share history and respond to
    each other, which is what produces the disagreement worth reading.
    """
    if not ag2_available():
        return None

    from autogen import AssistantAgent, GroupChat, GroupChatManager

    llm_config = build_llm_config(settings)
    sources = "\n\n".join(f"[{e.source_id}] {e.label}\n{e.text[:1500]}" for e in evidence[:8])

    critic = AssistantAgent(
        name="critic",
        llm_config=llm_config,
        system_message=(
            "You attack the draft. Find claims the cited passages do not actually support, "
            "figures that appear nowhere in the sources, and questions the draft dodges. "
            "Quote the passage you are checking against. Concede a point when the defender "
            "shows you the supporting text -- you are looking for real errors, not a win."
        ),
    )
    defender = AssistantAgent(
        name="defender",
        llm_config=llm_config,
        system_message=(
            "You defend the draft using only the supplied passages. When the critic is right, "
            "say so plainly and state the corrected claim. Never invent support that is not in "
            "the passages; conceding is the correct move when the evidence is not there."
        ),
    )

    chat = GroupChat(
        agents=[critic, defender],
        messages=[],
        max_round=max_turns,
        speaker_selection_method="round_robin",
    )
    manager = GroupChatManager(groupchat=chat, llm_config=llm_config)

    critic.initiate_chat(
        manager,
        message=(
            f"Question: {question}\n\n"
            f"Draft answer:\n{draft.draft_answer}\n\n"
            f"Claims:\n"
            + "\n".join(f"- {c.text} {c.citations}" for c in draft.claims)
            + f"\n\nSource passages:\n{sources}"
        ),
        clear_history=True,
    )
    return "\n\n".join(f"**{m.get('name', '?')}**: {m.get('content', '')}" for m in chat.messages)
