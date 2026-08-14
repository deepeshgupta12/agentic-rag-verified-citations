"""LLM client with schema-enforced outputs, retries and cost accounting.

Calling a chat endpoint once per step and hoping for JSON fails in three ways:
a raised call (bad key, rate limit, timeout) surfaces a raw traceback to the
user; a prose response yields ``{}`` and the pipeline continues on empty data;
and nothing records how many tokens or dollars a run consumed.

This wraps the provider with three guarantees the orchestrator relies on:

* **A validated object or an exception** -- never a silently empty dict.
* **Bounded retries** with backoff on transient failures, and a repair turn
  that shows the model its own invalid output plus the validation error.
* **Usage accounting** on every call, aggregated per run.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Sequence
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from . import jsonx
from .budget import Budget
from .config import Settings
from .schemas import Usage
from .tokens import count_tokens, estimate_cost

log = logging.getLogger("ragverify.llm")

T = TypeVar("T", bound=BaseModel)

# Errors worth retrying: transient server-side and network conditions.
_RETRYABLE = ("rate limit", "429", "timeout", "timed out", "500", "502", "503", "504",
              "overloaded", "connection", "temporarily")


class LLMError(RuntimeError):
    """Raised when a call cannot be completed after all retries."""


class SchemaError(LLMError):
    """Raised when the model never produced output matching the schema."""


def _is_retryable(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _RETRYABLE)


class LLMClient:
    """Thin wrapper over the OpenAI-compatible chat completions API.

    Kept provider-agnostic through ``base_url`` so the same code runs against
    OpenAI, Azure, OpenRouter, Together or a local vLLM/Ollama endpoint. The
    endpoint is selected by ``base_url`` alone.
    """

    def __init__(self, settings: Settings, client: Any = None, budget: Budget | None = None) -> None:
        self.settings = settings
        # Budget enforcement belongs at the call site, not at the loop
        # boundary. Checking only before an extra round leaves triage,
        # drafting, verification, repair turns, embeddings and synthesis
        # entirely unmetered -- a run configured for 1 call happily made 4.
        self.budget = budget
        self.usage = Usage()
        self._client = client or self._build_client(settings)
        # Set once we learn the model rejects response_format, so we stop
        # paying for a doomed first attempt on every subsequent call.
        self._structured_supported: bool | None = None

    @staticmethod
    def _build_client(settings: Settings) -> Any:
        from openai import OpenAI

        if not settings.api_key:
            raise LLMError("No API key configured. Set OPENAI_API_KEY or pass one in the UI.")
        kwargs: dict[str, Any] = {"api_key": settings.api_key, "timeout": settings.request_timeout_s * 4}
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        return OpenAI(**kwargs)

    # ------------------------------------------------------------------
    # Raw completion
    # ------------------------------------------------------------------

    def complete(
        self,
        system: str,
        user: str,
        *,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kwargs: dict[str, Any] = {"model": self.settings.model, "messages": messages}

        # Reasoning models reject an explicit temperature, and use a different
        # token-limit parameter name.
        if self.settings.supports_temperature():
            kwargs["temperature"] = self.settings.temperature
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
        elif max_tokens:
            kwargs["max_completion_tokens"] = max_tokens

        if response_format and self._structured_supported is not False:
            kwargs["response_format"] = response_format

        # Reserve before spending. A retry is a real request and must be
        # counted, so the check sits inside the attempt loop.
        last: Exception | None = None
        for attempt in range(self.settings.max_retries):
            if self.budget is not None:
                self.budget.check()
                # A fixed per-request timeout does not respect the run
                # deadline: a call started with 5s left could still run for
                # its full timeout and blow the cap.
                remaining = self.budget.remaining_seconds
                if remaining > 0:
                    kwargs["timeout"] = min(
                        kwargs.get("timeout", self.settings.request_timeout_s * 4), remaining
                    )
            try:
                response = self._client.chat.completions.create(**kwargs)
                self._record(response, system + user)
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:  # noqa: BLE001
                last = exc
                message = str(exc).lower()

                # Some gateways and older models 400 on response_format.
                # Drop it once and retry in plain mode rather than failing.
                if "response_format" in message or "json_schema" in message:
                    self._structured_supported = False
                    kwargs.pop("response_format", None)
                    continue

                if not _is_retryable(exc) or attempt == self.settings.max_retries - 1:
                    break

                # Exponential backoff with jitter, so parallel calls that were
                # rate limited together do not retry in lockstep.
                delay = (2**attempt) * 0.75 + random.uniform(0, 0.4)
                log.warning("LLM call failed (%s); retrying in %.1fs", exc, delay)
                time.sleep(delay)

        raise LLMError(f"LLM call failed after {self.settings.max_retries} attempts: {last}") from last

    def _record(self, response: Any, prompt_text: str) -> None:
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) or count_tokens(prompt_text)
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        self.usage = self.usage.add(
            Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=estimate_cost(prompt_tokens, completion_tokens, self.settings.price_per_million()),
                calls=1,
            )
        )
        self._sync_budget()

    def _sync_budget(self) -> None:
        """Mirror usage into the shared budget after every billed call."""
        if self.budget is not None:
            self.budget.calls = self.usage.calls
            self.budget.cost_usd = self.usage.cost_usd

    # ------------------------------------------------------------------
    # Structured output
    # ------------------------------------------------------------------

    def structured(
        self,
        system: str,
        user: str,
        schema: type[T],
        *,
        max_tokens: int | None = None,
        repair_attempts: int = 2,
    ) -> T:
        """Return an instance of ``schema`` or raise ``SchemaError``.

        Three layers, cheapest first: native ``json_schema`` response format;
        then tolerant extraction from prose; then a repair turn that feeds the
        model its own output and the validator's complaint.
        """
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": False,
                "schema": schema.model_json_schema(),
            },
        }
        instructions = f"{system}\n\nRespond with a single JSON object matching this schema:\n{schema.model_json_schema()}"

        raw = self.complete(instructions, user, response_format=response_format, max_tokens=max_tokens)
        parsed, error = self._validate(raw, schema)
        if parsed is not None:
            return parsed

        for _ in range(repair_attempts):
            repair_prompt = (
                f"Your previous response did not match the required schema.\n\n"
                f"Previous response:\n{raw[:2000]}\n\n"
                f"Validation error:\n{error}\n\n"
                f"Return ONLY corrected JSON. No prose, no code fences."
            )
            raw = self.complete(instructions, repair_prompt, response_format=response_format, max_tokens=max_tokens)
            parsed, error = self._validate(raw, schema)
            if parsed is not None:
                return parsed

        raise SchemaError(f"Could not obtain valid {schema.__name__}: {error}")

    @staticmethod
    def _validate(raw: str, schema: type[T]) -> tuple[T | None, str]:
        payload = jsonx.extract_object(raw)
        if payload is None:
            return None, "No JSON object found in the response."
        try:
            return schema.model_validate(payload), ""
        except ValidationError as exc:
            return None, str(exc)[:800]

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, texts: Sequence[str], batch_size: int = 96) -> list[list[float]]:
        """Embed ``texts``, batched to stay under request size limits."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            if self.budget is not None:
                self.budget.check()
            batch = [t or " " for t in texts[start : start + batch_size]]
            response = self._client.embeddings.create(model=self.settings.embed_model, input=batch)
            vectors.extend(item.embedding for item in response.data)
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            # Embeddings are billed and were being counted as free, so a run
            # that indexed a large corpus under-reported its own spend and
            # could pass a cost cap it had already exceeded.
            self.usage = self.usage.add(
                Usage(
                    prompt_tokens=prompt_tokens,
                    cost_usd=estimate_cost(prompt_tokens, 0, self.settings.embed_price_per_million()),
                    calls=1,
                )
            )
            self._sync_budget()
        return vectors
