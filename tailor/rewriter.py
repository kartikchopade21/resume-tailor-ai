"""Resume rewriting in three scopes, with an explicit skill->project routing step.

The routing step is the piece the pipeline is built around: given a skill the JD
demands that the resume never mentions, decide which existing project is the
most plausible home for it, and describe how that project would have used it.
A skill that has no plausible home is reported as unplaceable rather than
bolted onto an unrelated project.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz

from .ats import ATSReport
from .config import MODE_FULL_RESUME, MODE_PROJECTS_AND_SKILLS, MODE_SKILLS_ONLY
from .llm import complete_json
from .schema import Bullet, JobDescription, ResumeDoc

# A rewrite keeps the original's subject, domain and metrics, so it stays fairly
# close to its source. Set low enough to catch heavy rewordings, high enough that
# two genuinely different achievements are not treated as the same bullet.
_REWRITE_MATCH = 55.0

# --------------------------------------------------------------- shared rules
_HONESTY = """Hard constraints — these override every other instruction:
- Never invent an employer, job title, degree, certification, date, company name, \
or a metric that is not already in the resume.
- Never change dates, company names, institutions, or the candidate's identity.
- You may only reframe, expand, and re-word work the resume already describes.
- Every number you write must already appear in the resume. If a bullet needs a \
metric that is not there, write it without a metric rather than inventing one.
- Keep the candidate's own voice and level of seniority. No corporate filler.
- Bullets start with a strong past-tense verb, stay on one line (<= 30 words), \
and name the technology explicitly so a keyword parser sees it."""


@dataclass
class Assignment:
    skill: str
    target_type: str  # "project" | "experience"
    target_id: str
    target_name: str = ""
    rationale: str = ""
    how_applied: str = ""


@dataclass
class RewritePlan:
    assignments: list[Assignment] = field(default_factory=list)
    unplaceable: list[dict[str, str]] = field(default_factory=list)

    def for_target(self, target_id: str) -> list[Assignment]:
        return [a for a in self.assignments if a.target_id == target_id]


# ------------------------------------------------------------------- routing
_ROUTE_SYSTEM = f"""You route missing job-description skills to the resume item where \
each one most plausibly belongs.

For every skill you receive, choose the ONE project (preferred) or work-experience \
entry whose actual domain, tech stack and outcomes make it the most natural place \
for that skill to have been used. Judge by real technical adjacency, not by keyword \
similarity in the name.

Worked example: the JD requires "LLMs". The resume has (a) a React e-commerce site, \
(b) a customer-support ticket classifier built with scikit-learn, (c) a Django blog. \
The ticket classifier is the right home — it is already an NLP/text-classification \
system, so an LLM is a credible evolution of it ("replaced the TF-IDF classifier with \
an LLM-based intent classifier"). The e-commerce site is the wrong home even if it is \
the largest project.

If no item on the resume can host a skill without fabricating a new domain of work, \
put that skill in "unplaceable" with a one-line reason. That is the correct, expected \
answer for a genuine gap — do not force a bad fit.

{_HONESTY}

Return ONLY:
{{
  "assignments": [
    {{"skill": "...", "target_type": "project"|"experience", "target_id": "proj2",
      "rationale": "one sentence on why this item is the right home",
      "how_applied": "one concrete sentence describing how the skill was used in \
that specific item, referencing its real subject matter"}}
  ],
  "unplaceable": [{{"skill": "...", "reason": "..."}}]
}}"""


def route_skills(resume: ResumeDoc, jd: JobDescription, skills: list[str]) -> RewritePlan:
    """Decide which project/experience each missing skill belongs in."""
    plan = RewritePlan()
    if not skills:
        return plan

    inventory = {
        "projects": [
            {
                "id": p.id,
                "name": p.name,
                "subtitle": p.subtitle,
                "tech": p.tech,
                "bullets": [b.text for b in p.bullets],
            }
            for p in resume.projects
        ],
        "experience": [
            {
                "id": e.id,
                "title": e.title,
                "company": e.company,
                "dates": e.dates,
                "bullets": [b.text for b in e.bullets],
            }
            for e in resume.experience
        ],
    }
    user = (
        f"Target role: {jd.title}\n\n"
        f"Skills the resume is missing (route each one):\n"
        + "\n".join(f"- {s}" for s in skills)
        + "\n\nResume inventory:\n"
        + json.dumps(inventory, indent=2)
    )
    data = complete_json(_ROUTE_SYSTEM, user, max_tokens=12000, temperature=0.2)

    by_id = {p.id: p.name for p in resume.projects}
    by_id.update({e.id: f"{e.title} @ {e.company}" for e in resume.experience})
    valid_project_ids = {p.id for p in resume.projects}
    valid_exp_ids = {e.id for e in resume.experience}

    for item in data.get("assignments") or []:
        if not isinstance(item, dict):     # a malformed entry should not sink the round
            continue
        tid = str(item.get("target_id", ""))
        ttype = item.get("target_type", "project")
        if tid not in valid_project_ids and tid not in valid_exp_ids:
            plan.unplaceable.append(
                {"skill": item.get("skill", ""), "reason": "router returned an unknown target"}
            )
            continue
        plan.assignments.append(
            Assignment(
                skill=str(item.get("skill", "")).strip(),
                target_type="project" if tid in valid_project_ids else "experience",
                target_id=tid,
                target_name=by_id.get(tid, ""),
                rationale=str(item.get("rationale", "")).strip(),
                how_applied=str(item.get("how_applied", "")).strip(),
            )
        )
    for item in data.get("unplaceable") or []:
        if not isinstance(item, dict):
            continue
        plan.unplaceable.append(
            {"skill": str(item.get("skill", "")), "reason": str(item.get("reason", ""))}
        )
    return plan


# ------------------------------------------------------------------ rewriting
_SKILLS_ONLY_SYSTEM = f"""You update ONLY the skills section of a resume so it matches a \
job description's vocabulary.

- Add the requested skills into the most appropriate existing category, or create at \
most one new, conventionally named category (e.g. "AI/ML", "Cloud & DevOps").
- Where the resume already names a skill in a different surface form, rewrite it to \
the job description's form and keep the original in parentheses \
(e.g. "Large Language Models (LLMs)").
- Preserve every skill already present. Do not reorder categories gratuitously.
- Do not touch any other part of the resume.

{_HONESTY}

Return ONLY: {{"skills": {{"<Category>": ["skill", ...]}}}} — the complete new skills map."""


_APPLY_SYSTEM = f"""You rewrite specific resume items so that assigned skills are visibly \
demonstrated, then keep the skills section in sync.

For each item you are given a list of assigned skills, each with a "how_applied" note:
- Prefer rewriting an EXISTING bullet to surface the skill inside work already described. \
Mark those bullets origin="rewritten".
- Add a new bullet only when no existing bullet can carry the skill. Mark those \
origin="added". Add at most 2 new bullets per item, and keep each item at 5 bullets or fewer.
- Set "skills" on every bullet you touch to the assigned skills it demonstrates.
- Add the skill's technology names to the item's "tech" list as well.
- The rewritten bullet must stay true to what that project or job actually did: the \
subject matter, the domain and the outcome do not change — only the technical framing does.
- Leave untouched bullets exactly as they are, with origin="original".
- Return the COMPLETE bullet list for every item you include, in the original order, \
untouched bullets included verbatim. A bullet you omit is a real achievement deleted \
from the candidate's resume. Never drop, merge or summarise one, and never replace \
specific numbers, tools or outcomes with generic phrasing.

Also return the complete updated skills map with the assigned skills folded in.

{_HONESTY}

Return ONLY:
{{
  "projects": [{{"id":"proj1","name":"...","subtitle":"...","tech":["..."],
                 "bullets":[{{"text":"...","origin":"original|rewritten|added","skills":["..."]}}]}}],
  "experience": [{{"id":"exp1","title":"...","company":"...","dates":"...",
                   "bullets":[{{"text":"...","origin":"...","skills":["..."]}}]}}],
  "skills": {{"<Category>": ["..."]}}
}}
Include ONLY the projects and experience entries you actually changed."""


_FULL_SYSTEM = _APPLY_SYSTEM.replace(
    "Also return the complete updated skills map with the assigned skills folded in.",
    """Because the scope is the whole resume, also:
- Rewrite "headline" to the job description's exact job title when the candidate's real \
background supports it (e.g. "Machine Learning Engineer").
- Rewrite "summary" into 2-3 lines that lead with the target title, the candidate's real \
years/level, and the job's top required skills that the resume genuinely supports.
- You may additionally tighten any other experience or project bullet for keyword \
alignment, still under the honesty constraints.
- Return the complete updated skills map.""",
).replace(
    '  "skills": {"<Category>": ["..."]}\n}',
    '  "skills": {"<Category>": ["..."]},\n  "headline": "...",\n  "summary": "..."\n}',
)


def _apply_patch(resume: ResumeDoc, data: dict[str, Any], plan: RewritePlan) -> ResumeDoc:
    out = resume.copy()
    skill_by_target: dict[str, list[str]] = {}
    for a in plan.assignments:
        skill_by_target.setdefault(a.target_id, []).append(a.skill)

    projects = {p.id: p for p in out.projects}
    for patch in data.get("projects") or []:
        if not isinstance(patch, dict):
            continue
        proj = projects.get(str(patch.get("id", "")))
        if proj is None:
            continue
        if patch.get("name"):
            proj.name = str(patch["name"])
        if patch.get("subtitle"):
            proj.subtitle = str(patch["subtitle"])
        if patch.get("tech"):
            proj.tech = _merge_list(proj.tech, [str(t) for t in patch["tech"]])
        if patch.get("bullets"):
            proj.bullets = _merge_bullets(proj.bullets, patch["bullets"])
            _backfill_skills(proj.bullets, skill_by_target.get(proj.id, []))

    experiences = {e.id: e for e in out.experience}
    for patch in data.get("experience") or []:
        if not isinstance(patch, dict):
            continue
        exp = experiences.get(str(patch.get("id", "")))
        if exp is None:
            continue
        if patch.get("title"):
            exp.title = str(patch["title"])
        if patch.get("bullets"):
            exp.bullets = _merge_bullets(exp.bullets, patch["bullets"])
            _backfill_skills(exp.bullets, skill_by_target.get(exp.id, []))

    if isinstance(data.get("skills"), dict) and data["skills"]:
        merged: dict[str, list[str]] = {}
        for cat, vals in data["skills"].items():
            merged[str(cat)] = _merge_list([], [str(v) for v in (vals or [])])
        # never silently drop a skill the candidate already had
        existing = set(s.lower() for vals in merged.values() for s in vals)
        leftovers = [s for s in resume.all_skills() if s.lower() not in existing]
        if leftovers:
            merged.setdefault("Additional Skills", []).extend(leftovers)
        out.skills = merged

    if data.get("headline"):
        out.contact.headline = str(data["headline"]).strip()
    if data.get("summary"):
        out.summary = str(data["summary"]).strip()
    return out


def _merge_list(base: list[str], extra: list[str]) -> list[str]:
    seen, out = set(), []
    for item in list(base) + list(extra):
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def _merge_bullets(original: list[Bullet], returned: list[Any]) -> list[Bullet]:
    """Fold the model's bullets into the originals without losing any.

    The prompt asks for the complete bullet list back, but models routinely
    return only the lines they touched. Assigning that straight over the list
    deletes everything they left alone — which is how a role's real achievements
    disappear. So each original is matched to its rewrite by text similarity and
    replaced in place; anything unmatched is kept, and genuinely new bullets are
    appended. A missed match duplicates a line, which is visible and fixable;
    the alternative silently destroys the candidate's actual work.
    """
    incoming = [Bullet.coerce(b) for b in returned]
    incoming = [b for b in incoming if b.text.strip()]
    if not original:
        return incoming

    # "added" bullets are new work, never a rewrite of something already there
    candidates = [b for b in incoming if b.origin != "added"]
    used: set[int] = set()
    merged: list[Bullet] = []

    for orig in original:
        best, best_score = None, 0.0
        for i, cand in enumerate(candidates):
            if i in used:
                continue
            score = fuzz.token_set_ratio(orig.text, cand.text)
            if score > best_score:
                best, best_score = i, score
        if best is not None and best_score >= _REWRITE_MATCH:
            used.add(best)
            merged.append(candidates[best])
        else:
            merged.append(orig)          # untouched by the model — keep it verbatim

    matched = {id(candidates[i]) for i in used}
    merged.extend(b for b in incoming if id(b) not in matched and b.origin == "added")
    merged.extend(
        b for b in incoming
        if id(b) not in matched and b.origin != "added" and b not in merged
    )
    return merged


def _ensure_skills_listed(resume: ResumeDoc, required: list[str]) -> None:
    """Guarantee every must-have JD skill appears in the skills section.

    Routing is the model's judgement call, so a required skill can come back
    unplaceable or simply be dropped from a long list. The ATS scorer already
    reports anything listed but not demonstrated as a weak skill, so listing it
    here is visible rather than silent — and the alternative, losing a mandatory
    keyword outright, costs the candidate the screen.
    """
    if not required:
        return
    present = {s.strip().lower() for s in resume.all_skills() if s.strip()}
    missing = [s for s in required if s.strip() and s.strip().lower() not in present]
    if not missing:
        return
    bucket = next(
        (c for c in resume.skills if c.strip().lower() in ("additional skills", "other")),
        "Additional Skills",
    )
    resume.skills[bucket] = _merge_list(resume.skills.get(bucket, []), missing)


def _backfill_skills(bullets: list[Bullet], skills: list[str]) -> None:
    for b in bullets:
        if b.origin in ("added", "rewritten") and not b.skills:
            b.skills = list(skills)


def rewrite(
    resume: ResumeDoc,
    jd: JobDescription,
    report: ATSReport,
    mode: str,
    feedback: str = "",
) -> tuple[ResumeDoc, RewritePlan]:
    """One rewrite round. Returns the new resume and the routing plan used."""
    missing = report.missing_required + report.missing_preferred
    weak = report.weak_skills

    if mode == MODE_SKILLS_ONLY:
        wanted = _merge_list(missing, report.missing_keywords[:12])
        if not wanted:
            return resume.copy(), RewritePlan()
        user = (
            f"Target role: {jd.title}\n"
            f"Skills to add: {', '.join(wanted)}\n\n"
            f"Current skills section:\n{json.dumps(resume.skills, indent=2)}\n\n"
            f"ATS feedback:\n{feedback or report.gap_summary()}"
        )
        data = complete_json(_SKILLS_ONLY_SYSTEM, user, max_tokens=8000, temperature=0.2)
        out = _apply_patch(resume, {"skills": data.get("skills") or {}}, RewritePlan())
        _ensure_skills_listed(out, jd.must_have_skills)
        return out, RewritePlan()

    # modes 2 and 3 both route skills into real work first
    to_route = _merge_list(missing, weak)
    plan = route_skills(resume, jd, to_route) if to_route else RewritePlan()

    targets: list[dict[str, Any]] = []
    for proj in resume.projects:
        assigned = plan.for_target(proj.id)
        if not assigned and mode != MODE_FULL_RESUME:
            continue
        targets.append(
            {
                "type": "project", "id": proj.id, "name": proj.name,
                "subtitle": proj.subtitle, "tech": proj.tech,
                "bullets": [b.text for b in proj.bullets],
                "assigned_skills": [
                    {"skill": a.skill, "how_applied": a.how_applied,
                     "rationale": a.rationale} for a in assigned
                ],
            }
        )
    for exp in resume.experience:
        assigned = plan.for_target(exp.id)
        if not assigned and mode != MODE_FULL_RESUME:
            continue
        targets.append(
            {
                "type": "experience", "id": exp.id, "title": exp.title,
                "company": exp.company, "dates": exp.dates,
                "bullets": [b.text for b in exp.bullets],
                "assigned_skills": [
                    {"skill": a.skill, "how_applied": a.how_applied,
                     "rationale": a.rationale} for a in assigned
                ],
            }
        )

    if not targets:
        return resume.copy(), plan

    system = _FULL_SYSTEM if mode == MODE_FULL_RESUME else _APPLY_SYSTEM
    user = (
        f"Target role: {jd.title} ({jd.seniority or 'unspecified level'})\n"
        f"Job's required skills: {', '.join(jd.must_have_skills)}\n"
        f"Job's preferred skills: {', '.join(jd.nice_to_have_skills + jd.tools)}\n"
        f"ATS keywords still missing: {', '.join(report.missing_keywords[:20])}\n\n"
        f"Current headline: {resume.contact.headline or '(none)'}\n"
        f"Current summary: {resume.summary or '(none)'}\n"
        f"Current skills:\n{json.dumps(resume.skills, indent=2)}\n\n"
        f"Items to update:\n{json.dumps(targets, indent=2)}\n\n"
        f"ATS gap report:\n{feedback or report.gap_summary()}"
    )
    data = complete_json(system, user, max_tokens=16000, temperature=0.3)
    out = _apply_patch(resume, data, plan)
    _ensure_skills_listed(out, jd.must_have_skills)
    return out, plan
