"""Provider-agnostic JSON completion: prompt in, validated JSON out.

Open-weight models are markedly worse than frontier ones at emitting strict
JSON on demand, so this layer does the work to make them usable: native JSON
mode where the provider supports it, brace-matched extraction, an explicit
repair round trip, and backoff on the rate limits that free tiers enforce.
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from . import providers
from .config import SETTINGS


class LLMError(RuntimeError):
    pass


class RateLimited(LLMError):
    pass


class Truncated(LLMError):
    """The model ran out of output budget mid-answer, so the JSON is incomplete.

    Reasoning models bill their hidden thinking against max_tokens, so a budget
    that looks generous can leave nothing for the answer itself. Retrying the
    same request unchanged just truncates again — the budget has to grow.
    """


_CLIENT = None
_CLIENT_SIG: tuple[str, str, str] = ("", "", "")


def reset_client() -> None:
    """Drop the cached client so a new key or provider takes effect."""
    global _CLIENT, _CLIENT_SIG
    _CLIENT, _CLIENT_SIG = None, ("", "", "")


def active_provider() -> providers.Provider:
    return providers.get(SETTINGS.provider)


def _client():
    global _CLIENT, _CLIENT_SIG
    provider = active_provider()
    key = SETTINGS.api_key or provider.key_from_env()

    if provider.needs_key and not key:
        raise LLMError(
            f"No API key for {provider.label}. Get one free at {provider.console_url}, "
            f"then paste it in the sidebar or set {provider.env_var} in .env."
        )

    signature = (provider.key, key, SETTINGS.model)
    if _CLIENT is not None and _CLIENT_SIG == signature:
        return _CLIENT

    if provider.protocol == "anthropic":
        from anthropic import Anthropic

        _CLIENT = Anthropic(api_key=key)
    else:
        from openai import OpenAI

        _CLIENT = OpenAI(
            api_key=key or "not-needed",       # Ollama ignores it but requires a value
            base_url=provider.base_url,
            timeout=180.0,
            max_retries=0,                     # we do our own backoff
        )
    _CLIENT_SIG = signature
    return _CLIENT


# ------------------------------------------------------------------ parsing
def _extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of a model response."""
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    # some open models prefix a reasoning preamble or a <think> block
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise LLMError(f"Model did not return valid JSON. Got:\n{text[:600]}")


def _as_object(data: Any) -> dict:
    """Callers expect the documented single JSON object.

    Models — Gemini especially — sometimes wrap it in a one-element array, or
    split it into several objects. Unwrap those; anything else is a schema miss,
    so raise and let the caller's repair retry ask again.
    """
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        objects = [d for d in data if isinstance(d, dict)]
        if len(objects) == 1:
            return objects[0]
        if objects:                        # keys are disjoint in every schema here
            merged: dict = {}
            for obj in objects:
                merged.update(obj)
            return merged
    raise LLMError(
        f"Model returned {type(data).__name__}, expected a JSON object. "
        f"Got:\n{json.dumps(data)[:600]}"
    )


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "rate limit" in text or "429" in text or "quota" in text or "tpm" in text


# ------------------------------------------------------------------- calling
def _call_anthropic(system: str, user: str, max_tokens: int, temperature: float) -> str:
    resp = _client().messages.create(
        model=SETTINGS.model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[
            {"role": "user", "content": user},
            {"role": "assistant", "content": "{"},  # prefill straight into JSON
        ],
    )
    text = "{" + "".join(
        b.text for b in resp.content if getattr(b, "type", "") == "text"
    )
    if resp.stop_reason == "max_tokens":
        raise Truncated(f"Answer cut off at max_tokens={max_tokens}. Ends: …{text[-80:]!r}")
    return text


def _call_openai_compatible(
    system: str, user: str, max_tokens: int, temperature: float, json_mode: bool
) -> str:
    kwargs: dict[str, Any] = {
        "model": SETTINGS.model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    effort = active_provider().reasoning_effort
    if effort:
        # spend the budget on the answer, not on hidden thinking
        kwargs["reasoning_effort"] = effort
    try:
        resp = _client().chat.completions.create(**kwargs)
    except Exception as exc:
        # not every model on every gateway accepts JSON mode — retry without it
        text = str(exc)
        if json_mode and ("response_format" in text or "json_object" in text):
            kwargs.pop("response_format")
            resp = _client().chat.completions.create(**kwargs)
        elif effort and "reasoning_effort" in text:
            kwargs.pop("reasoning_effort")
            resp = _client().chat.completions.create(**kwargs)
        else:
            raise

    choice = resp.choices[0]
    content = choice.message.content or ""
    if choice.finish_reason == "length":
        raise Truncated(
            f"Answer cut off at max_tokens={max_tokens} "
            f"(returned {len(content)} chars). Ends: …{content[-80:]!r}"
        )
    return content


def complete_json(
    system: str,
    user: str,
    *,
    max_tokens: int = 8000,
    temperature: float = 0.2,
    retries: int = 3,
) -> Any:
    """Call the active provider and parse a JSON response."""
    provider = active_provider()
    ceiling = provider.max_output_tokens
    max_tokens = min(max_tokens, ceiling)
    # open models drift without an explicit instruction, even in JSON mode
    system = system + "\n\nRespond with a single valid JSON object and nothing else."

    prompt = user
    last_err: Exception | None = None

    for attempt in range(retries + 1):
        try:
            if provider.protocol == "anthropic":
                raw = _call_anthropic(system, prompt, max_tokens, temperature)
            else:
                raw = _call_openai_compatible(
                    system, prompt, max_tokens, temperature, provider.supports_json_mode
                )
            return _as_object(_extract_json(raw))

        except Truncated as exc:           # out of room — reruns need a bigger budget
            last_err = exc
            if max_tokens >= ceiling:
                raise Truncated(
                    f"{exc} Already at this provider's ceiling of {ceiling} tokens — "
                    f"ask for less in one call, or raise max_output_tokens for "
                    f"{provider.key} in providers.py."
                ) from exc
            max_tokens = min(max_tokens * 2, ceiling)
            prompt = (
                user
                + "\n\nYour previous reply was cut off before the JSON closed. Keep every "
                "field, but write terse values — one short sentence each, no padding."
            )

        except LLMError as exc:            # parsed badly — ask for a repair
            last_err = exc
            prompt = (
                user
                + "\n\nYour previous reply was not valid JSON. Reply with the JSON "
                "object only — no prose, no markdown fences, no commentary."
            )
            temperature = max(0.0, temperature - 0.1)

        except Exception as exc:           # transport, rate limit, overload
            last_err = exc
            if attempt >= retries:
                if _is_rate_limit(exc):
                    raise RateLimited(
                        f"{provider.label} rate limit hit. Free tiers are metered per "
                        f"minute — wait a moment and retry, or switch provider. ({exc})"
                    ) from exc
                raise
            delay = (4.0 if _is_rate_limit(exc) else 1.5) * (attempt + 1)
            time.sleep(delay + random.uniform(0, 0.5))

    raise LLMError(str(last_err))
