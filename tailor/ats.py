"""Deterministic ATS scorer.

No third-party ATS API is used: none of the well-known checkers (Jobscan,
Resume Worded, Enhancv, LiveCareer, Resume.io) expose a free public scoring
endpoint. This module reimplements the rubric those tools describe publicly so
the score is free, unlimited, reproducible, and — critically — differentiable
enough to steer the rewrite loop.

Scoring is intentionally strict about *where* a skill appears: a keyword that
only shows up in the skills list scores less than one demonstrated inside a
project or job bullet, which is exactly what real recruiters and the better
parsers weight for.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

from rapidfuzz import fuzz

from .config import SETTINGS, Weights
from .parser import FormatAudit
from .schema import JobDescription, ResumeDoc

# --------------------------------------------------------------------- setup
FUZZ_THRESHOLD = 88

# Bidirectional alias groups. Anything in a group matches anything else in it.
ALIAS_GROUPS: list[list[str]] = [
    ["llm", "llms", "large language model", "large language models"],
    ["genai", "gen ai", "generative ai"],
    ["nlp", "natural language processing"],
    ["ml", "machine learning"],
    ["dl", "deep learning"],
    ["cv", "computer vision"],
    ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    ["js", "javascript"],
    ["ts", "typescript"],
    ["py", "python"],
    ["k8s", "kubernetes"],
    ["ci/cd", "ci cd", "cicd", "continuous integration", "continuous deployment"],
    ["aws", "amazon web services"],
    ["gcp", "google cloud", "google cloud platform"],
    ["azure", "microsoft azure"],
    ["postgres", "postgresql"],
    ["mongo", "mongodb"],
    ["rest", "rest api", "restful", "restful api", "rest apis"],
    ["api", "apis"],
    ["sql", "structured query language"],
    ["tf", "tensorflow"],
    ["sklearn", "scikit-learn", "scikit learn"],
    ["hf", "hugging face", "huggingface"],
    ["db", "database", "databases"],
    ["oop", "object oriented programming", "object-oriented programming"],
    ["ui", "user interface"],
    ["ux", "user experience"],
    ["qa", "quality assurance"],
    ["etl", "extract transform load", "data pipeline", "data pipelines"],
    ["mlops", "ml ops", "machine learning operations"],
    ["vector db", "vector database", "vector databases", "vector store"],
    ["prompt engineering", "prompting", "prompt design"],
    ["fine tuning", "fine-tuning", "finetuning"],
    ["react", "react.js", "reactjs"],
    ["node", "node.js", "nodejs"],
    ["ai", "artificial intelligence"],
]

_STOPWORDS = {
    "and", "or", "the", "a", "an", "of", "for", "to", "in", "on", "with", "by",
    "as", "at", "from", "is", "are", "be", "our", "your", "you", "we", "will",
    "that", "this", "their", "its", "it", "using", "use", "used", "strong",
    "experience", "knowledge", "ability", "skills", "work", "working", "etc",
    "including", "such", "other", "new", "across", "into", "them", "they",
}


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9+#./\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokens(text: str) -> list[str]:
    return [t for t in normalize(text).split() if t]


def _stem(token: str) -> str:
    """Crude suffix stripping so pipeline/pipelines/pipelining collapse together."""
    for suffix in ("ingly", "edly", "ings", "ing", "ies", "ied", "ers", "er", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            base = token[: -len(suffix)]
            if suffix == "ies":
                base += "y"
            return base
    return token


def _content_tokens(text: str) -> set[str]:
    return {
        _stem(t)
        for t in _split_joiners(normalize(text)).split()
        if t not in _STOPWORDS and len(t) > 2
    }


def _variants(term: str) -> set[str]:
    """The term plus its alias-group siblings plus a simple plural/singular."""
    base = normalize(term)
    out = {base}
    for group in ALIAS_GROUPS:
        if base in group:
            out.update(group)
    for v in list(out):
        if v.endswith("ies") and len(v) > 4:
            out.add(v[:-3] + "y")
        elif v.endswith("es") and len(v) > 4:
            out.add(v[:-2])
        if v.endswith("s") and len(v) > 3:
            out.add(v[:-1])
        else:
            out.add(v + "s")
    return {v for v in out if v}


def _split_joiners(text: str) -> str:
    """`llm-based` -> `llm based`, `ci/cd` -> `ci cd`, `node.js` -> `node js`.

    Indexed alongside the unsplit form so both `scikit-learn` and `scikit learn`
    match, and so a keyword glued to a suffix ("LLM-powered") is still found.
    """
    return re.sub(r"\s+", " ", re.sub(r"[-/.]", " ", text)).strip()


class TextIndex:
    """Fast containment + fuzzy matching over a body of text."""

    def __init__(self, text: str):
        self.norm = normalize(text)
        self.split = _split_joiners(self.norm)
        self.padded = f" {self.norm} "
        self.padded_split = f" {self.split} "
        self.ngrams: dict[int, set[str]] = {}
        for source in (self.norm, self.split):
            toks = source.split()
            for n in range(1, 5):
                grams = {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}
                self.ngrams.setdefault(n, set()).update(grams)

    def contains(self, term: str) -> bool:
        for variant in _variants(term):
            if not variant:
                continue
            if f" {variant} " in self.padded:
                return True
            split_variant = _split_joiners(variant)
            if split_variant and f" {split_variant} " in self.padded_split:
                return True
            for candidate_term in {variant, split_variant}:
                if len(candidate_term) < 5:
                    continue
                pool = self.ngrams.get(min(len(candidate_term.split()), 4))
                if not pool:
                    continue
                for candidate in pool:
                    if abs(len(candidate) - len(candidate_term)) <= 4 and (
                        fuzz.ratio(candidate, candidate_term) >= FUZZ_THRESHOLD
                    ):
                        return True
        return False


# -------------------------------------------------------------------- result
@dataclass
class SkillHit:
    skill: str
    required: bool
    found: bool
    # "evidence" (project/experience bullet), "skills_list", "elsewhere", "none"
    location: str = "none"
    credit: float = 0.0


@dataclass
class ScoreBreakdown:
    hard_skills: float = 0.0
    keyword_coverage: float = 0.0
    title_match: float = 0.0
    responsibilities: float = 0.0
    parseability: float = 0.0
    completeness: float = 0.0

    def total(self) -> float:
        return round(sum(self.__dict__.values()), 1)


@dataclass
class ATSReport:
    score: float = 0.0
    breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    max_breakdown: dict[str, float] = field(default_factory=dict)
    skill_hits: list[SkillHit] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    missing_preferred: list[str] = field(default_factory=list)
    weak_skills: list[str] = field(default_factory=list)   # listed but never demonstrated
    missing_keywords: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    uncovered_responsibilities: list[str] = field(default_factory=list)
    format_issues: list[str] = field(default_factory=list)
    completeness_issues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def gap_summary(self) -> str:
        """Compact feedback string handed back to the rewriter each round."""
        lines = [f"Current ATS score: {self.score}/100 (target {SETTINGS.target_score})."]
        for label, value in asdict(self.breakdown).items():
            cap = self.max_breakdown.get(label, 0)
            if value < cap - 0.05:
                lines.append(f"- {label}: {round(value, 1)}/{cap}")
        if self.missing_required:
            lines.append("MISSING REQUIRED SKILLS: " + ", ".join(self.missing_required))
        if self.missing_preferred:
            lines.append("MISSING PREFERRED SKILLS: " + ", ".join(self.missing_preferred))
        if self.weak_skills:
            lines.append(
                "LISTED BUT NEVER DEMONSTRATED (move these into a project or job bullet): "
                + ", ".join(self.weak_skills)
            )
        if self.missing_keywords:
            lines.append("MISSING ATS KEYWORDS: " + ", ".join(self.missing_keywords[:20]))
        if self.uncovered_responsibilities:
            lines.append(
                "RESPONSIBILITIES WITH NO MATCHING BULLET: "
                + " | ".join(self.uncovered_responsibilities[:6])
            )
        if self.completeness_issues:
            lines.append("MISSING FROM RESUME: " + ", ".join(self.completeness_issues))
        if self.format_issues:
            lines.append("FORMAT: " + " ".join(self.format_issues))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["breakdown"] = asdict(self.breakdown)
        return d


# --------------------------------------------------------------------- score
def score_resume(
    resume: ResumeDoc,
    jd: JobDescription,
    audit: FormatAudit | None = None,
    weights: Weights | None = None,
) -> ATSReport:
    w = weights or SETTINGS.weights
    report = ATSReport()
    report.max_breakdown = {
        "hard_skills": w.hard_skills,
        "keyword_coverage": w.keyword_coverage,
        "title_match": w.title_match,
        "responsibilities": w.responsibilities,
        "parseability": w.parseability,
        "completeness": w.completeness,
    }

    # Three indexes so we can tell *where* a term appears.
    evidence_text = " \n ".join(
        [e.text() for e in resume.experience] + [p.text() for p in resume.projects]
    )
    evidence_idx = TextIndex(evidence_text)
    skills_idx = TextIndex(" , ".join(resume.all_skills()))
    whole_idx = TextIndex(resume.full_text())

    # --- 1. hard skills -----------------------------------------------------
    required = _dedupe(jd.must_have_skills)
    preferred = _dedupe(jd.nice_to_have_skills + jd.tools, exclude=required)
    weighted_total = 2.0 * len(required) + 1.0 * len(preferred)
    earned = 0.0

    for skill, is_req in [(s, True) for s in required] + [(s, False) for s in preferred]:
        weight = 2.0 if is_req else 1.0
        if evidence_idx.contains(skill):
            hit = SkillHit(skill, is_req, True, "evidence", 1.0)
        elif skills_idx.contains(skill):
            hit = SkillHit(skill, is_req, True, "skills_list", 0.7)
        elif whole_idx.contains(skill):
            hit = SkillHit(skill, is_req, True, "elsewhere", 0.85)
        else:
            hit = SkillHit(skill, is_req, False, "none", 0.0)
        earned += weight * hit.credit
        report.skill_hits.append(hit)
        if not hit.found:
            (report.missing_required if is_req else report.missing_preferred).append(skill)
        elif hit.location == "skills_list":
            report.weak_skills.append(skill)

    report.breakdown.hard_skills = (
        w.hard_skills * (earned / weighted_total) if weighted_total else w.hard_skills
    )

    # --- 2. verbatim keyword coverage --------------------------------------
    keywords = _dedupe(jd.ats_keywords)
    if keywords:
        for kw in keywords:
            (report.matched_keywords if whole_idx.contains(kw)
             else report.missing_keywords).append(kw)
        report.breakdown.keyword_coverage = w.keyword_coverage * (
            len(report.matched_keywords) / len(keywords)
        )
    else:
        report.breakdown.keyword_coverage = w.keyword_coverage

    # --- 3. job title match -------------------------------------------------
    report.breakdown.title_match = _title_score(resume, jd, w.title_match, report)

    # --- 4. responsibility coverage ----------------------------------------
    resps = [r for r in jd.responsibilities if r.strip()]
    if resps:
        # Partial credit, not a threshold: a JD line is never echoed word for word,
        # and a smooth signal is what lets the rewrite loop climb.
        FULL_CREDIT_OVERLAP = 0.5
        ev_tokens = _content_tokens(evidence_text)
        covered = 0.0
        for resp in resps:
            need = _content_tokens(resp)
            if not need:
                covered += 1.0
                continue
            overlap = len(need & ev_tokens) / len(need)
            credit = min(1.0, overlap / FULL_CREDIT_OVERLAP)
            covered += credit
            if credit < 0.75:
                report.uncovered_responsibilities.append(resp)
        report.breakdown.responsibilities = w.responsibilities * (covered / len(resps))
    else:
        report.breakdown.responsibilities = w.responsibilities

    # --- 5. parseability ----------------------------------------------------
    report.breakdown.parseability = _parseability(resume, audit, w.parseability, report)

    # --- 6. completeness ----------------------------------------------------
    report.breakdown.completeness = _completeness(resume, w.completeness, report)

    report.score = report.breakdown.total()
    if report.score >= SETTINGS.target_score:
        report.notes.append("Meets the target score.")
    return report


def _dedupe(items: Iterable[str], exclude: Iterable[str] = ()) -> list[str]:
    blocked = {normalize(x) for x in exclude}
    seen, out = set(), []
    for item in items:
        key = normalize(item)
        if not key or key in seen or key in blocked:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


def _title_score(resume: ResumeDoc, jd: JobDescription, cap: float, report: ATSReport) -> float:
    title = normalize(jd.title)
    if not title:
        return cap
    surfaces = [resume.contact.headline, resume.summary] + [
        e.title for e in resume.experience
    ] + [p.name for p in resume.projects]
    best = 0.0
    for surface in surfaces:
        s = normalize(surface)
        if not s:
            continue
        if title in s:
            return cap
        best = max(best, fuzz.token_set_ratio(title, s) / 100.0)
    core = {t for t in title.split() if t not in _STOPWORDS}
    if core:
        surf_tokens = _content_tokens(" ".join(surfaces))
        best = max(best, len(core & surf_tokens) / len(core))
    if best < 0.99:
        report.notes.append(
            f'Job title "{jd.title}" is not stated verbatim in the resume headline.'
        )
    return cap * min(1.0, best)


def _parseability(resume: ResumeDoc, audit: FormatAudit | None, cap: float,
                  report: ATSReport) -> float:
    score = cap
    penalties = [
        (audit.has_tables if audit else False, 3.0, "Tables in the source file."),
        (audit.has_images if audit else False, 2.0, "Images in the source file."),
        (audit.has_text_boxes if audit else False, 3.0, "Text boxes in the source file."),
        (audit.multi_column if audit else False, 3.0, "Multi-column layout."),
        (audit.in_header_footer if audit else False, 2.0, "Content in header/footer."),
        (audit.non_standard_bullets if audit else False, 1.0, "Decorative bullet glyphs."),
        (bool(audit.unusual_fonts) if audit else False, 1.0, "Non-standard fonts."),
        ((audit.page_count > 2) if audit else False, 1.0, "More than 2 pages."),
    ]
    for triggered, penalty, msg in penalties:
        if triggered:
            score -= penalty
            report.format_issues.append(msg)

    raw = (resume.raw_text or resume.full_text()).lower()
    expected = {
        "experience": ("experience", "employment", "work history", "professional"),
        "education": ("education", "academic"),
        "skills": ("skills", "technical skills", "competencies"),
    }
    for label, variants in expected.items():
        if not any(v in raw for v in variants):
            score -= 1.5
            report.format_issues.append(f'No recognizable "{label}" section heading.')

    words = len((resume.raw_text or resume.full_text()).split())
    if words < 300:
        score -= 2.0
        report.format_issues.append(f"Thin resume ({words} words); ATS keyword density suffers.")
    elif words > 1200:
        score -= 1.0
        report.format_issues.append(f"Very long resume ({words} words).")
    return max(0.0, score)


def _completeness(resume: ResumeDoc, cap: float, report: ATSReport) -> float:
    c = resume.contact
    checks = [
        (bool(c.email and "@" in c.email), 3.0, "email address"),
        (bool(c.phone and sum(ch.isdigit() for ch in c.phone) >= 7), 2.0, "phone number"),
        (bool(c.location), 1.0, "location"),
        (bool(c.linkedin or c.github or c.portfolio), 1.5, "LinkedIn/GitHub/portfolio link"),
        (bool(resume.summary.strip()), 1.0, "professional summary"),
        (bool(resume.education), 1.0, "education section"),
        (bool(resume.all_skills()), 0.5, "skills section"),
    ]
    total = sum(pts for _, pts, _ in checks)
    earned = 0.0
    for ok, pts, label in checks:
        if ok:
            earned += pts
        else:
            report.completeness_issues.append(label)
    return cap * (earned / total)
