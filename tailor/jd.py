"""Job description -> structured requirements."""
from __future__ import annotations

from .llm import complete_json
from .schema import JobDescription

_JD_SYSTEM = """You extract hiring requirements from a job description for an ATS \
keyword-matching engine.

Return ONLY a JSON object:
{
  "title": "the exact job title",
  "company": "",
  "seniority": "intern|junior|mid|senior|lead|manager|",
  "must_have_skills": ["..."],
  "nice_to_have_skills": ["..."],
  "tools": ["named products, frameworks, platforms, libraries"],
  "soft_skills": ["..."],
  "responsibilities": ["short verb-first phrases"],
  "qualifications": ["degrees, years of experience, certifications"],
  "ats_keywords": ["..."]
}

Rules for skills and keywords:
- Use the job description's own surface form. If it says "LLMs", emit "LLMs"; if it \
says "Large Language Models", emit that. Do not translate between them.
- must_have_skills: required/essential ones. nice_to_have_skills: preferred/bonus ones. \
A skill belongs to exactly one of the two lists.
- Skills must be concrete and matchable: "PyTorch", "RAG pipelines", "prompt engineering", \
"CI/CD". Never vague phrases like "strong background" or "modern tooling".
- ats_keywords: 15-30 terms an ATS would literally string-match, including the job \
title, key skills, and domain nouns. Single words or short noun phrases only.
- Deduplicate case-insensitively across all lists. Omit nothing that is stated; \
invent nothing that is not."""


def parse_jd(text: str) -> JobDescription:
    text = (text or "").strip()
    if len(text.split()) < 20:
        raise ValueError("The job description is too short to extract requirements from.")
    data = complete_json(
        _JD_SYSTEM,
        f"Job description:\n<jd>\n{text}\n</jd>",
        max_tokens=4000,
        temperature=0.0,
    )
    jd = JobDescription.from_dict(data)
    jd.raw_text = text
    if not jd.hard_skills():
        raise ValueError("No concrete skills could be extracted from this job description.")
    return jd
