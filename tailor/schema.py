"""Data model shared by every stage of the pipeline.

The resume is carried as structured JSON rather than raw text so the rewriter
can target one project without disturbing the rest of the document, and so the
renderer can emit ATS-safe DOCX deterministically.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Contact:
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    github: str = ""
    portfolio: str = ""
    headline: str = ""  # e.g. "Machine Learning Engineer"


@dataclass
class Bullet:
    text: str
    # provenance: "original" | "rewritten" | "added"
    origin: str = "original"
    # which JD skill(s) motivated this bullet, for the review panel
    skills: list[str] = field(default_factory=list)

    @classmethod
    def coerce(cls, value: Any) -> "Bullet":
        if isinstance(value, Bullet):
            return value
        if isinstance(value, str):
            return cls(text=value)
        if isinstance(value, dict):
            return cls(
                text=str(value.get("text", "")).strip(),
                origin=value.get("origin", "original"),
                skills=list(value.get("skills", []) or []),
            )
        return cls(text=str(value))


@dataclass
class Project:
    id: str = ""
    name: str = ""
    subtitle: str = ""          # role / context / dates
    tech: list[str] = field(default_factory=list)
    bullets: list[Bullet] = field(default_factory=list)
    link: str = ""

    def text(self) -> str:
        return " ".join(
            [self.name, self.subtitle, " ".join(self.tech)]
            + [b.text for b in self.bullets]
        )


@dataclass
class Experience:
    id: str = ""
    company: str = ""
    title: str = ""
    location: str = ""
    dates: str = ""
    bullets: list[Bullet] = field(default_factory=list)

    def text(self) -> str:
        return " ".join(
            [self.company, self.title, self.location, self.dates]
            + [b.text for b in self.bullets]
        )


@dataclass
class Education:
    institution: str = ""
    degree: str = ""
    dates: str = ""
    details: str = ""


@dataclass
class ResumeDoc:
    contact: Contact = field(default_factory=Contact)
    summary: str = ""
    skills: dict[str, list[str]] = field(default_factory=dict)  # category -> skills
    experience: list[Experience] = field(default_factory=list)
    projects: list[Project] = field(default_factory=list)
    education: list[Education] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    extras: dict[str, list[str]] = field(default_factory=dict)  # awards, publications...
    # raw text of the uploaded file, used for parseability checks
    raw_text: str = ""
    source_format: str = ""  # pdf | docx | txt

    # ---------------------------------------------------------------- helpers
    def all_skills(self) -> list[str]:
        out: list[str] = []
        for values in self.skills.values():
            out.extend(values)
        return out

    def full_text(self) -> str:
        """Everything an ATS would index, as one blob."""
        parts = [
            self.contact.name,
            self.contact.headline,
            self.contact.email,
            self.contact.phone,
            self.contact.location,
            self.contact.linkedin,
            self.contact.github,
            self.summary,
        ]
        for cat, vals in self.skills.items():
            parts.append(cat)
            parts.extend(vals)
        for exp in self.experience:
            parts.append(exp.text())
        for proj in self.projects:
            parts.append(proj.text())
        for edu in self.education:
            parts.extend([edu.institution, edu.degree, edu.dates, edu.details])
        parts.extend(self.certifications)
        for vals in self.extras.values():
            parts.extend(vals)
        return "\n".join(p for p in parts if p)

    def added_claims(self) -> list[dict[str, Any]]:
        """Every bullet the pipeline wrote or rewrote, for human review."""
        claims: list[dict[str, Any]] = []
        for proj in self.projects:
            for b in proj.bullets:
                if b.origin != "original":
                    claims.append(
                        {"where": f"Project · {proj.name}", "origin": b.origin,
                         "skills": b.skills, "text": b.text}
                    )
        for exp in self.experience:
            for b in exp.bullets:
                if b.origin != "original":
                    claims.append(
                        {"where": f"Experience · {exp.title} @ {exp.company}",
                         "origin": b.origin, "skills": b.skills, "text": b.text}
                    )
        return claims

    # ------------------------------------------------------------ (de)serial.
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResumeDoc":
        doc = cls()
        c = data.get("contact") or {}
        doc.contact = Contact(**{k: c.get(k, "") or "" for k in Contact().__dict__})
        doc.summary = data.get("summary", "") or ""

        skills = data.get("skills") or {}
        if isinstance(skills, list):  # tolerate a flat list
            skills = {"Skills": skills}
        doc.skills = {
            str(k): [str(v) for v in (vals or [])] for k, vals in skills.items()
        }

        for i, e in enumerate(data.get("experience") or []):
            doc.experience.append(
                Experience(
                    id=e.get("id") or f"exp{i + 1}",
                    company=e.get("company", "") or "",
                    title=e.get("title", "") or "",
                    location=e.get("location", "") or "",
                    dates=e.get("dates", "") or "",
                    bullets=[Bullet.coerce(b) for b in (e.get("bullets") or [])],
                )
            )
        for i, p in enumerate(data.get("projects") or []):
            doc.projects.append(
                Project(
                    id=p.get("id") or f"proj{i + 1}",
                    name=p.get("name", "") or "",
                    subtitle=p.get("subtitle", "") or "",
                    tech=[str(t) for t in (p.get("tech") or [])],
                    bullets=[Bullet.coerce(b) for b in (p.get("bullets") or [])],
                    link=p.get("link", "") or "",
                )
            )
        for ed in data.get("education") or []:
            doc.education.append(
                Education(
                    institution=ed.get("institution", "") or "",
                    degree=ed.get("degree", "") or "",
                    dates=ed.get("dates", "") or "",
                    details=ed.get("details", "") or "",
                )
            )
        doc.certifications = [str(x) for x in (data.get("certifications") or [])]
        doc.extras = {
            str(k): [str(v) for v in (vals or [])]
            for k, vals in (data.get("extras") or {}).items()
        }
        doc.raw_text = data.get("raw_text", "") or ""
        doc.source_format = data.get("source_format", "") or ""
        return doc

    def copy(self) -> "ResumeDoc":
        return ResumeDoc.from_dict(json.loads(self.to_json()))


@dataclass
class JobDescription:
    title: str = ""
    company: str = ""
    seniority: str = ""
    must_have_skills: list[str] = field(default_factory=list)
    nice_to_have_skills: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    soft_skills: list[str] = field(default_factory=list)
    responsibilities: list[str] = field(default_factory=list)
    qualifications: list[str] = field(default_factory=list)
    # verbatim surface forms an ATS is likely to string-match on
    ats_keywords: list[str] = field(default_factory=list)
    raw_text: str = ""

    def hard_skills(self) -> list[str]:
        seen, out = set(), []
        for s in self.must_have_skills + self.nice_to_have_skills + self.tools:
            k = s.strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append(s.strip())
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobDescription":
        jd = cls()
        for key in jd.__dict__:
            if key in data and data[key] is not None:
                setattr(jd, key, data[key])
        return jd
