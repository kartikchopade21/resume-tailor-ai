"""The pipeline: parse -> score -> rewrite -> rescore, until the target or a plateau."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .ats import ATSReport, score_resume
from .config import SETTINGS, MODE_PROJECTS_AND_SKILLS
from .jd import parse_jd
from .parser import FormatAudit, load_resume, load_resume_bytes
from .render import to_plain_text
from .rewriter import RewritePlan, rewrite
from .schema import JobDescription, ResumeDoc

ProgressFn = Callable[[str], None]


@dataclass
class Iteration:
    index: int
    score: float
    report: ATSReport
    resume: ResumeDoc
    plan: RewritePlan | None = None


@dataclass
class PipelineResult:
    original: ResumeDoc
    original_report: ATSReport
    final: ResumeDoc
    final_report: ATSReport
    jd: JobDescription
    mode: str
    iterations: list[Iteration] = field(default_factory=list)
    target_met: bool = False
    stop_reason: str = ""
    unplaceable: list[dict[str, str]] = field(default_factory=list)
    assignments: list[dict[str, str]] = field(default_factory=list)

    @property
    def delta(self) -> float:
        return round(self.final_report.score - self.original_report.score, 1)

    def blocking_gaps(self) -> list[str]:
        """Why the target was not reached, in plain language."""
        gaps: list[str] = []
        r = self.final_report
        for item in self.unplaceable:
            gaps.append(
                f"{item['skill']} — no project or role on the resume can host it honestly"
                + (f" ({item['reason']})" if item.get("reason") else "")
            )
        for skill in r.missing_required:
            if not any(skill == g.split(" — ")[0] for g in gaps):
                gaps.append(f"{skill} — required by the JD, still absent")
        for issue in r.completeness_issues:
            gaps.append(f"Resume has no {issue}")
        for issue in r.format_issues[:3]:
            gaps.append(issue)
        return gaps


def _clean_audit(resume: ResumeDoc) -> FormatAudit:
    """The audit that applies to *our* rendered output, not the uploaded file."""
    text = to_plain_text(resume)
    words = len(text.split())
    audit = FormatAudit(word_count=words, page_count=1 if words < 700 else 2)
    return audit


def run_pipeline(
    resume_source: str | tuple[bytes, str],
    jd_text: str,
    mode: str = MODE_PROJECTS_AND_SKILLS,
    target_score: float | None = None,
    max_iterations: int | None = None,
    progress: ProgressFn | None = None,
) -> PipelineResult:
    """resume_source is a path, or (file_bytes, filename)."""
    target = target_score if target_score is not None else SETTINGS.target_score
    max_iters = max_iterations if max_iterations is not None else SETTINGS.max_iterations
    say: ProgressFn = progress or (lambda _msg: None)

    say("Parsing resume…")
    if isinstance(resume_source, tuple):
        resume, audit = load_resume_bytes(resume_source[0], resume_source[1])
    else:
        resume, audit = load_resume(resume_source)

    say("Extracting job requirements…")
    jd = parse_jd(jd_text)

    say("Scoring the original resume…")
    original_report = score_resume(resume, jd, audit)
    original = resume.copy()

    result = PipelineResult(
        original=original,
        original_report=original_report,
        final=original,
        final_report=original_report,
        jd=jd,
        mode=mode,
    )
    result.iterations.append(Iteration(0, original_report.score, original_report, original))

    if original_report.score >= target:
        result.target_met = True
        result.stop_reason = "The original resume already meets the target."
        return result

    current, current_report = resume, original_report
    best, best_report = original, original_report
    feedback = original_report.gap_summary()
    seen_unplaceable: dict[str, str] = {}

    for i in range(1, max_iters + 1):
        say(f"Round {i}/{max_iters}: rewriting…")
        try:
            candidate, plan = rewrite(current, jd, current_report, mode, feedback)
        except Exception as exc:  # a failed round should not lose earlier progress
            result.stop_reason = f"Rewrite failed on round {i}: {exc}"
            break

        for item in plan.unplaceable:
            if item.get("skill"):
                seen_unplaceable.setdefault(item["skill"], item.get("reason", ""))
        for a in plan.assignments:
            # a later round may re-route the same skill; keep the most recent decision
            entry = {
                "skill": a.skill,
                "target": a.target_name or a.target_id,
                "target_type": a.target_type,
                "rationale": a.rationale,
                "how_applied": a.how_applied,
            }
            result.assignments = [
                x for x in result.assignments if x["skill"].lower() != a.skill.lower()
            ]
            result.assignments.append(entry)

        # from here on, parseability reflects the clean document we render
        candidate.raw_text = to_plain_text(candidate)
        say(f"Round {i}/{max_iters}: rescoring…")
        cand_report = score_resume(candidate, jd, _clean_audit(candidate))
        result.iterations.append(Iteration(i, cand_report.score, cand_report, candidate, plan))

        improvement = cand_report.score - current_report.score
        if cand_report.score > best_report.score:
            best, best_report = candidate, cand_report

        if cand_report.score >= target:
            result.target_met = True
            result.stop_reason = f"Target reached on round {i}."
            current, current_report = candidate, cand_report
            break

        if improvement < SETTINGS.plateau_delta and i >= 2:
            result.stop_reason = (
                f"Stopped on round {i}: the score plateaued at {best_report.score} "
                "— the remaining gaps need real experience, not rewording."
            )
            current, current_report = candidate, cand_report
            break

        current, current_report = candidate, cand_report
        feedback = cand_report.gap_summary()
    else:
        result.stop_reason = (
            f"Stopped after {max_iters} rounds at {best_report.score} without "
            f"reaching {target}."
        )

    result.final = best
    result.final_report = best_report
    result.unplaceable = [{"skill": k, "reason": v} for k, v in seen_unplaceable.items()]
    if result.final_report.score >= target:
        result.target_met = True
    return result


def diff_summary(before: ResumeDoc, after: ResumeDoc) -> dict[str, Any]:
    """What actually changed, for the review panel."""
    changes: dict[str, Any] = {"skills_added": [], "bullets": [], "headline": None,
                               "summary": None}
    before_skills = {s.lower() for s in before.all_skills()}
    changes["skills_added"] = [
        s for s in after.all_skills() if s.lower() not in before_skills
    ]
    if before.contact.headline != after.contact.headline:
        changes["headline"] = (before.contact.headline, after.contact.headline)
    if before.summary != after.summary:
        changes["summary"] = (before.summary, after.summary)

    before_bullets = {
        **{p.id: [b.text for b in p.bullets] for p in before.projects},
        **{e.id: [b.text for b in e.bullets] for e in before.experience},
    }
    for item, label in (
        [(p, f"Project · {p.name}") for p in after.projects]
        + [(e, f"Experience · {e.title} @ {e.company}") for e in after.experience]
    ):
        old = before_bullets.get(item.id, [])
        for b in item.bullets:
            if b.origin == "original" and b.text in old:
                continue
            changes["bullets"].append(
                {
                    "where": label,
                    "origin": b.origin if b.origin != "original" else "changed",
                    "skills": b.skills,
                    "text": b.text,
                    "was": _closest(old, b.text),
                }
            )
    return changes


def _closest(candidates: list[str], text: str) -> str:
    if not candidates:
        return ""
    from rapidfuzz import process, fuzz

    match = process.extractOne(text, candidates, scorer=fuzz.token_set_ratio)
    return match[0] if match and match[1] >= 45 else ""
