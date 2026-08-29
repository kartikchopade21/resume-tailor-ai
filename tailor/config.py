"""Central configuration for the resume tailoring pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:  # optional
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


# --- Rewrite modes (the dropdown) -------------------------------------------
MODE_SKILLS_ONLY = "skills_only"
MODE_PROJECTS_AND_SKILLS = "projects_and_skills"
MODE_FULL_RESUME = "full_resume"

MODE_LABELS = {
    MODE_SKILLS_ONLY: "Add it in skills only",
    MODE_PROJECTS_AND_SKILLS: "Add it in projects and skills",
    MODE_FULL_RESUME: "Add overall in the resume",
}

MODE_HELP = {
    MODE_SKILLS_ONLY: (
        "Only the Skills section is touched. Fastest and safest, but the lowest "
        "ceiling on ATS score because keywords have no supporting evidence."
    ),
    MODE_PROJECTS_AND_SKILLS: (
        "For every missing JD skill the pipeline picks the project it fits best "
        "and rewrites that project to show how the skill was actually applied, "
        "then mirrors the skill into the Skills section."
    ),
    MODE_FULL_RESUME: (
        "Everything is in scope: headline, summary, skills, projects and work "
        "experience bullets are all realigned to the job description."
    ),
}


# --- ATS scoring rubric weights (sum = 100) ---------------------------------
@dataclass(frozen=True)
class Weights:
    hard_skills: float = 40.0        # must-have skills weigh 2x nice-to-have
    keyword_coverage: float = 20.0   # verbatim ATS keyword surface forms
    title_match: float = 10.0        # job title alignment
    responsibilities: float = 5.0    # action/responsibility overlap
    parseability: float = 15.0       # formatting an ATS can actually read
    completeness: float = 10.0       # contact info + expected sections


def _default_provider() -> str:
    from . import providers

    return providers.detect_from_env()


def _default_model(provider: str) -> str:
    from . import providers

    return os.getenv("TAILOR_MODEL", "") or providers.get(provider).default_model


def _default_key(provider: str) -> str:
    from . import providers

    return providers.get(provider).key_from_env()


@dataclass
class Settings:
    provider: str = field(default_factory=_default_provider)
    api_key: str = ""
    model: str = ""
    target_score: float = field(
        default_factory=lambda: float(os.getenv("TAILOR_TARGET_SCORE", "90"))
    )
    max_iterations: int = field(
        default_factory=lambda: int(os.getenv("TAILOR_MAX_ITERATIONS", "5"))
    )
    # stop early if two consecutive rounds move the score less than this
    plateau_delta: float = 1.0
    weights: Weights = field(default_factory=Weights)

    def __post_init__(self) -> None:
        if not self.model:
            self.model = _default_model(self.provider)
        if not self.api_key:
            self.api_key = _default_key(self.provider)

    def use_provider(self, name: str, api_key: str | None = None,
                     model: str | None = None) -> None:
        """Switch provider, resetting the model and key to that provider's defaults."""
        from . import llm, providers

        provider = providers.get(name)
        self.provider = provider.key
        self.model = model or os.getenv("TAILOR_MODEL", "") or provider.default_model
        self.api_key = api_key if api_key is not None else provider.key_from_env()
        llm.reset_client()

    @property
    def configured(self) -> bool:
        from . import providers

        return bool(self.api_key) or not providers.get(self.provider).needs_key


SETTINGS = Settings()
