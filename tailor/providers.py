"""Model providers.

Everything except Anthropic speaks the OpenAI chat-completions protocol, so one
client covers Groq, OpenRouter, Ollama, Gemini and OpenAI — only the base URL,
the key and the default model change.

Free options, in the order worth trying:
  groq        free key, open weights (Llama 3.3 70B), fastest inference
  openrouter  free key, large open models (GLM, Nemotron) via ":free" model ids
  ollama      no key at all, fully local and offline, no rate limits
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    key: str
    label: str
    default_model: str
    env_var: str | None            # where the key is read from
    base_url: str | None           # None = the SDK's own default
    protocol: str                  # "anthropic" | "openai"
    free: bool
    needs_key: bool
    console_url: str
    notes: str
    # free tiers are token-stingy; cap our own requests to stay inside them
    max_output_tokens: int = 8000
    supports_json_mode: bool = True
    # reasoning models bill hidden thinking against max_tokens; keeping it low
    # leaves the budget for the answer. "" = don't send the parameter at all.
    reasoning_effort: str = ""

    def key_from_env(self) -> str:
        return os.getenv(self.env_var, "") if self.env_var else ""


PROVIDERS: dict[str, Provider] = {
    "groq": Provider(
        key="groq",
        label="Groq — free, open weights (recommended)",
        default_model="llama-3.3-70b-versatile",
        env_var="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        protocol="openai",
        free=True,
        needs_key=True,
        console_url="https://console.groq.com/keys",
        notes="Free tier is roughly 30 requests/min and 1,000/day, with a tight "
              "per-minute token budget — fine for tailoring a handful of resumes, "
              "and the fastest inference of the free options.",
        max_output_tokens=4000,
    ),
    "openrouter": Provider(
        key="openrouter",
        label="OpenRouter — free, large open models",
        default_model="z-ai/glm-5.2:free",
        env_var="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        protocol="openai",
        free=True,
        needs_key=True,
        console_url="https://openrouter.ai/keys",
        notes="Any model id ending in ':free' costs nothing. Bigger models than "
              "Groq's, but slower and with a lower daily request cap.",
        max_output_tokens=8000,
    ),
    "ollama": Provider(
        key="ollama",
        label="Ollama — fully local, no key, no limits",
        default_model="qwen2.5:14b-instruct",
        env_var=None,
        base_url=os.getenv("OLLAMA_HOST_URL", "http://localhost:11434/v1"),
        protocol="openai",
        free=True,
        needs_key=False,
        console_url="https://ollama.com/download",
        notes="Runs on your own machine — nothing leaves it and there is no quota. "
              "On 16GB of RAM a 14B model fits comfortably. Slower, and smaller "
              "models are the least reliable at the structured JSON this pipeline "
              "asks for, so expect the occasional retry.",
        max_output_tokens=8000,
    ),
    "gemini": Provider(
        key="gemini",
        label="Google Gemini — free tier (closed weights)",
        # free-tier quota is 20 requests/day *per model*, so switching the model
        # in the sidebar gets you a fresh allowance once one is exhausted
        default_model="gemini-3.1-flash-lite",
        env_var="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        protocol="openai",
        free=True,
        needs_key=True,
        console_url="https://aistudio.google.com/apikey",
        notes="Generous free tier and strong instruction-following, but the model "
              "itself is not open source.",
        # Gemini 3.x thinks before answering and charges that thinking to the
        # output budget, so a routing reply needs far more headroom than its
        # visible JSON suggests.
        max_output_tokens=32000,
        reasoning_effort="low",
    ),
    "anthropic": Provider(
        key="anthropic",
        label="Anthropic Claude — paid",
        default_model="claude-sonnet-4-6",
        env_var="ANTHROPIC_API_KEY",
        base_url=None,
        protocol="anthropic",
        free=False,
        needs_key=True,
        console_url="https://platform.claude.com/settings/keys",
        notes="Best quality on the skill-routing judgement. Costs a few cents per run.",
        max_output_tokens=8000,
    ),
    "openai": Provider(
        key="openai",
        label="OpenAI — paid",
        default_model="gpt-4o-mini",
        env_var="OPENAI_API_KEY",
        base_url=None,
        protocol="openai",
        free=False,
        needs_key=True,
        console_url="https://platform.openai.com/api-keys",
        notes="Pay per token.",
        max_output_tokens=8000,
    ),
}

FREE_PROVIDERS = [p for p in PROVIDERS.values() if p.free]
DEFAULT_PROVIDER = "groq"


def get(name: str) -> Provider:
    provider = PROVIDERS.get((name or "").strip().lower())
    if provider is None:
        raise ValueError(
            f"Unknown provider {name!r}. Choose one of: {', '.join(PROVIDERS)}"
        )
    return provider


def detect_from_env() -> str:
    """Pick a provider based on which key is actually present."""
    explicit = os.getenv("TAILOR_PROVIDER", "").strip().lower()
    if explicit in PROVIDERS:
        return explicit
    for name in ("groq", "openrouter", "anthropic", "gemini", "openai"):
        if PROVIDERS[name].key_from_env():
            return name
    return DEFAULT_PROVIDER
