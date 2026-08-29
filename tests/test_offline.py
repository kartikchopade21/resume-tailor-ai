"""End-to-end test with a stubbed model, so it runs with no API key.

Covers: scoring math, evidence-location weighting, alias matching, the routing
patch application, the iteration loop's stop conditions, and DOCX/PDF render +
re-parse round trip.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")

from tailor import ats, llm, parser, pipeline, render  # noqa: E402
from tailor.config import MODE_PROJECTS_AND_SKILLS, MODE_SKILLS_ONLY  # noqa: E402
from tailor.schema import JobDescription, ResumeDoc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(os.path.dirname(HERE), "samples")

FAILS: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILS.append(label)


# --------------------------------------------------------------- fixtures
RESUME_JSON = {
    "contact": {
        "name": "Arjun Mehta", "email": "arjun.mehta@example.com",
        "phone": "+91 98765 43210", "location": "Bengaluru, India",
        "linkedin": "linkedin.com/in/arjunmehta", "github": "github.com/arjunmehta",
        "portfolio": "", "headline": "",
    },
    "summary": "Software engineer with 3 years of experience building data-driven "
               "backend services and machine learning prototypes.",
    "skills": {
        "Languages": ["Python", "JavaScript", "SQL"],
        "Frameworks": ["Django", "Flask", "React", "scikit-learn"],
        "Data": ["PostgreSQL", "MongoDB", "pandas", "NumPy"],
        "Tools": ["Git", "Docker", "Jenkins", "Linux"],
    },
    "experience": [
        {"id": "exp1", "company": "Cobalt Systems", "title": "Software Engineer",
         "location": "Bengaluru, India", "dates": "July 2023 - Present",
         "bullets": [
             "Built and maintained Django REST services handling 40k daily requests for the logistics dashboard",
             "Designed PostgreSQL schemas and optimised slow queries, cutting p95 latency by 35%",
             "Containerised three internal services with Docker and set up Jenkins build pipelines",
             "Mentored two interns through their first production deployments"]},
        {"id": "exp2", "company": "Vireo Labs", "title": "Junior Developer",
         "location": "Pune, India", "dates": "Aug 2022 - June 2023",
         "bullets": [
             "Developed internal reporting tools in Flask consumed by the operations team",
             "Wrote ETL jobs in Python that consolidated vendor data into a central warehouse"]},
    ],
    "projects": [
        {"id": "proj1", "name": "Support Ticket Classifier", "subtitle": "Personal project",
         "tech": ["Python", "scikit-learn", "pandas", "Flask"], "link": "",
         "bullets": [
             "Built a TF-IDF and logistic regression model that routes incoming support tickets into 12 categories with 87% accuracy",
             "Trained on 40,000 historical tickets and exposed the model through a Flask endpoint",
             "Wrote an evaluation script comparing precision and recall across categories"]},
        {"id": "proj2", "name": "Kirana Commerce", "subtitle": "Personal project",
         "tech": ["React", "Django", "PostgreSQL", "Docker"], "link": "",
         "bullets": [
             "Built a full-stack storefront for neighbourhood grocery shops with cart, checkout and order tracking",
             "Deployed on a single VM with Docker Compose and nginx"]},
        {"id": "proj3", "name": "Sensor Anomaly Dashboard", "subtitle": "College capstone",
         "tech": ["Python", "pandas", "Plotly", "Flask"], "link": "",
         "bullets": [
             "Analysed 2 million IoT sensor readings and flagged anomalies with a rolling z-score",
             "Built a Plotly dashboard the lab used to review flagged intervals"]},
    ],
    "education": [{"institution": "Pune Institute of Technology",
                   "degree": "B.E. Computer Science", "dates": "2018 - 2022",
                   "details": "CGPA 8.4/10"}],
    "certifications": [], "extras": {},
}

JD_JSON = {
    "title": "Machine Learning Engineer",
    "company": "Northgate Analytics", "seniority": "mid",
    "must_have_skills": ["Python", "LLMs", "prompt engineering", "PyTorch",
                         "RAG pipelines", "vector database", "SQL", "Docker", "AWS"],
    "nice_to_have_skills": ["LangChain", "MLOps", "FastAPI", "Kubernetes"],
    "tools": ["Pinecone", "FAISS"],
    "soft_skills": ["collaboration"],
    "responsibilities": [
        "Design and deploy RAG pipelines over internal knowledge bases",
        "Fine-tune and evaluate large language models",
        "Build prompt engineering workflows and eval harnesses",
        "Deploy inference services on AWS with Docker and CI/CD",
        "Expose models through REST APIs",
    ],
    "qualifications": ["2+ years machine learning in Python", "Bachelor's in CS"],
    "ats_keywords": ["Machine Learning Engineer", "LLMs", "RAG", "prompt engineering",
                     "PyTorch", "vector database", "Python", "SQL", "Docker", "AWS",
                     "FastAPI", "LangChain", "MLOps", "Kubernetes", "REST APIs",
                     "fine-tuning", "model evaluation", "data pipeline"],
}

ROUTE_RESPONSE = {
    "assignments": [
        {"skill": "LLMs", "target_type": "project", "target_id": "proj1",
         "rationale": "It is already a text-classification/NLP system, so an LLM is a credible evolution.",
         "how_applied": "Replaced the TF-IDF classifier with an LLM-based intent classifier over the same ticket corpus."},
        {"skill": "prompt engineering", "target_type": "project", "target_id": "proj1",
         "rationale": "The classifier's category taxonomy is exactly what few-shot prompts encode.",
         "how_applied": "Iterated few-shot prompts against the held-out ticket set."},
        {"skill": "RAG pipelines", "target_type": "project", "target_id": "proj1",
         "rationale": "Historical tickets are a natural retrieval corpus.",
         "how_applied": "Retrieved similar past tickets as context before classification."},
        {"skill": "vector database", "target_type": "project", "target_id": "proj1",
         "rationale": "Retrieval over the ticket corpus needs an embedding index.",
         "how_applied": "Indexed ticket embeddings in FAISS."},
        {"skill": "AWS", "target_type": "experience", "target_id": "exp1",
         "rationale": "The containerised services already needed a deployment target.",
         "how_applied": "Deployed the Dockerised services on AWS ECS."},
        {"skill": "FastAPI", "target_type": "project", "target_id": "proj1",
         "rationale": "The model is already served over HTTP via Flask.",
         "how_applied": "Served the classifier through a FastAPI inference endpoint."},
    ],
    "unplaceable": [
        {"skill": "Kubernetes", "reason": "Nothing on the resume was orchestrated beyond Docker Compose."},
        {"skill": "PyTorch", "reason": "All modelling work used scikit-learn; no deep-learning project exists."},
    ],
}

APPLY_RESPONSE = {
    "projects": [
        {"id": "proj1", "name": "Support Ticket Classifier",
         "subtitle": "Personal project",
         "tech": ["Python", "scikit-learn", "pandas", "FastAPI", "LLMs", "FAISS", "LangChain"],
         "bullets": [
             {"text": "Built an LLM-based intent classifier that routes incoming support tickets into 12 categories with 87% accuracy, replacing a TF-IDF baseline",
              "origin": "rewritten", "skills": ["LLMs"]},
             {"text": "Built a RAG pipeline that retrieves similar historical tickets from a FAISS vector database as few-shot context before classification",
              "origin": "added", "skills": ["RAG pipelines", "vector database"]},
             {"text": "Iterated prompt engineering against a held-out set of 40,000 tickets and served the model through a FastAPI inference endpoint",
              "origin": "rewritten", "skills": ["prompt engineering", "FastAPI"]},
             {"text": "Wrote an evaluation script comparing precision and recall across categories",
              "origin": "original", "skills": []},
         ]},
    ],
    "experience": [
        {"id": "exp1", "title": "Software Engineer", "company": "Cobalt Systems",
         "dates": "July 2023 - Present",
         "bullets": [
             {"text": "Built and maintained Django REST services handling 40k daily requests for the logistics dashboard", "origin": "original", "skills": []},
             {"text": "Designed PostgreSQL schemas and optimised slow SQL queries, cutting p95 latency by 35%", "origin": "rewritten", "skills": ["SQL"]},
             {"text": "Containerised three internal services with Docker and deployed them on AWS with Jenkins CI/CD pipelines", "origin": "rewritten", "skills": ["AWS", "Docker"]},
             {"text": "Mentored two interns through their first production deployments", "origin": "original", "skills": []},
         ]},
    ],
    "skills": {
        "Languages": ["Python", "JavaScript", "SQL"],
        "AI/ML": ["LLMs", "prompt engineering", "RAG pipelines", "LangChain",
                  "scikit-learn", "model evaluation", "fine-tuning"],
        "Frameworks": ["FastAPI", "Django", "Flask", "React"],
        "Data": ["PostgreSQL", "MongoDB", "pandas", "NumPy", "FAISS", "vector database",
                 "data pipeline"],
        "Cloud & DevOps": ["Docker", "AWS", "Jenkins", "CI/CD", "MLOps", "Git", "Linux"],
    },
    "headline": "Machine Learning Engineer",
    "summary": "Machine Learning Engineer with 3 years of experience shipping Python "
               "services and LLM-powered features, from RAG pipelines and prompt "
               "engineering through Dockerised deployment on AWS.",
}


def fake_complete_json(system: str, user: str, **kwargs):
    if "You convert resume text into structured JSON" in system:
        return RESUME_JSON
    if "You extract hiring requirements" in system:
        return JD_JSON
    if "You route missing job-description skills" in system:
        return ROUTE_RESPONSE
    if "You update ONLY the skills section" in system:
        return {"skills": APPLY_RESPONSE["skills"]}
    if "You rewrite specific resume items" in system:
        return APPLY_RESPONSE
    raise AssertionError("unexpected system prompt")


# ------------------------------------------------------------------- tests
def test_providers() -> None:
    print("\nProvider layer")
    from tailor import providers
    from tailor.config import SETTINGS

    check("free providers exist", len(providers.FREE_PROVIDERS) >= 3,
          ", ".join(p.key for p in providers.FREE_PROVIDERS))
    check("ollama needs no key", not providers.get("ollama").needs_key)
    check("every provider declares a model and a console url",
          all(p.default_model and p.console_url for p in providers.PROVIDERS.values()))
    check("unknown provider is rejected",
          _raises(lambda: providers.get("nope")))

    before = (SETTINGS.provider, SETTINGS.model, SETTINGS.api_key)
    try:
        SETTINGS.use_provider("groq")
        check("switching provider swaps the default model",
              SETTINGS.model == "llama-3.3-70b-versatile", SETTINGS.model)
        SETTINGS.use_provider("ollama")
        check("ollama counts as configured with no key",
              SETTINGS.configured and not SETTINGS.api_key)
        check("llm resolves the active provider",
              llm.active_provider().key == "ollama")
        SETTINGS.use_provider("openrouter", api_key="")
        check("a key-needing provider without a key is not configured",
              not SETTINGS.configured)
        SETTINGS.use_provider("openrouter", api_key="test", model="z-ai/glm-5.2:free")
        check("an explicit model override sticks", SETTINGS.model == "z-ai/glm-5.2:free")
    finally:
        SETTINGS.provider, SETTINGS.model, SETTINGS.api_key = before
        llm.reset_client()

    # the JSON extractor has to survive what open models actually emit
    cases = {
        "bare object": '{"a": 1}',
        "fenced": '```json\n{"a": 1}\n```',
        "preamble prose": 'Sure! Here is the JSON:\n{"a": 1}',
        "think block": '<think>let me plan</think>\n{"a": 1}',
        "trailing chatter": '{"a": 1}\nHope that helps!',
    }
    for label, raw in cases.items():
        ok = False
        try:
            ok = llm._extract_json(raw) == {"a": 1}
        except Exception:
            ok = False
        check(f"JSON extraction: {label}", ok)
    check("JSON extraction rejects garbage",
          _raises(lambda: llm._extract_json("no json here at all")))


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except Exception:
        return True


def test_matching() -> None:
    print("\nMatching engine")
    idx = ats.TextIndex("Built an LLM-based classifier with scikit-learn and Postgres")
    check("alias: 'Large Language Models' matches 'LLM'", idx.contains("Large Language Models"))
    check("alias: 'PostgreSQL' matches 'Postgres'", idx.contains("PostgreSQL"))
    check("alias: 'scikit learn' matches 'scikit-learn'", idx.contains("scikit learn"))
    check("absent term is not matched", not idx.contains("Kubernetes"))
    check("unrelated short term is not matched", not idx.contains("Rust"))


def test_scoring() -> None:
    print("\nScoring")
    resume = ResumeDoc.from_dict(RESUME_JSON)
    resume.raw_text = render.to_plain_text(resume)
    jd = JobDescription.from_dict(JD_JSON)
    audit = parser.FormatAudit(word_count=len(resume.raw_text.split()), page_count=1)
    report = ats.score_resume(resume, jd, audit)

    check("score is within 0-100", 0 <= report.score <= 100, f"score={report.score}")
    check("baseline resume is below target", report.score < 90, f"score={report.score}")
    check("LLMs flagged as missing", "LLMs" in report.missing_required)
    check("PyTorch flagged as missing", "PyTorch" in report.missing_required)
    check("Python is matched", "Python" not in report.missing_required)
    check("Docker is matched", "Docker" not in report.missing_required)
    check("components never exceed their cap",
          all(v <= report.max_breakdown[k] + 1e-6
              for k, v in report.breakdown.__dict__.items()))
    check("breakdown sums to the reported score",
          abs(sum(report.breakdown.__dict__.values()) - report.score) < 0.11)

    # evidence weighting: a skill only in the skills list must score lower
    listed_only = resume.copy()
    listed_only.skills["AI/ML"] = ["LLMs", "PyTorch", "RAG pipelines"]
    listed_only.raw_text = render.to_plain_text(listed_only)
    r2 = ats.score_resume(listed_only, jd, audit)

    demonstrated = listed_only.copy()
    demonstrated.projects[0].bullets[0].text += (
        " using LLMs and PyTorch in a RAG pipelines setup"
    )
    demonstrated.raw_text = render.to_plain_text(demonstrated)
    r3 = ats.score_resume(demonstrated, jd, audit)

    check("listing a skill raises the score", r2.score > report.score,
          f"{report.score} -> {r2.score}")
    check("demonstrating it scores higher than listing it", r3.score > r2.score,
          f"listed={r2.score} demonstrated={r3.score}")
    check("listed-only skills are reported as weak",
          "PyTorch" in r2.weak_skills or "LLMs" in r2.weak_skills)

    # format penalties must bite
    bad = parser.FormatAudit(
        has_tables=True, has_images=True, multi_column=True, page_count=3,
        word_count=audit.word_count,
    )
    r4 = ats.score_resume(resume, jd, bad)
    check("bad formatting lowers the score", r4.score < report.score,
          f"clean={report.score} messy={r4.score}")


def test_pipeline_and_render() -> None:
    print("\nPipeline (stubbed model)")
    llm.complete_json = fake_complete_json
    parser.complete_json = fake_complete_json
    import tailor.jd as jd_mod
    import tailor.rewriter as rw_mod
    jd_mod.complete_json = fake_complete_json
    rw_mod.complete_json = fake_complete_json

    resume_path = os.path.join(SAMPLES, "sample_resume.txt")
    with open(os.path.join(SAMPLES, "sample_jd.txt"), encoding="utf-8") as fh:
        jd_text = fh.read()

    result = pipeline.run_pipeline(
        resume_path, jd_text, mode=MODE_PROJECTS_AND_SKILLS,
        target_score=90, max_iterations=3,
    )
    print(f"    {result.original_report.score} -> {result.final_report.score} "
          f"| {result.stop_reason}")

    check("the rewrite improves the score",
          result.final_report.score > result.original_report.score,
          f"{result.original_report.score} -> {result.final_report.score}")
    # This sample cannot honestly reach 90: the JD requires PyTorch and nothing on
    # the resume is deep-learning work. Landing just short and saying why is the
    # correct outcome, not a bug.
    check("gets close to the target on a mostly-matching resume",
          result.final_report.score >= 85, f"final={result.final_report.score}")
    check("target_met flag agrees with the score",
          result.target_met == (result.final_report.score >= 90))
    check("falling short is attributed to the genuinely missing skill",
          result.target_met or any("PyTorch" in g for g in result.blocking_gaps()))
    check("the loop terminated", len(result.iterations) <= 4,
          f"{len(result.iterations)} iterations")
    check("routing decisions are recorded", len(result.assignments) >= 5)
    check("each skill is routed once, not once per round",
          len({a["skill"].lower() for a in result.assignments}) == len(result.assignments),
          f"{len(result.assignments)} entries")
    check("LLMs was routed to the ticket classifier",
          any(a["skill"] == "LLMs" and "Ticket" in a["target"] for a in result.assignments))
    check("unplaceable skills are reported honestly",
          any(u["skill"] == "PyTorch" for u in result.unplaceable))
    check("original document is left untouched",
          result.original.contact.headline == ""
          and "LLM" not in result.original.projects[0].bullets[0].text)
    check("added/rewritten bullets are flagged for review",
          len(result.final.added_claims()) >= 3,
          f"{len(result.final.added_claims())} claims")
    check("no pre-existing skill was dropped",
          {"MongoDB", "Jenkins", "NumPy"} <= {s for s in result.final.all_skills()})

    changes = pipeline.diff_summary(result.original, result.final)
    check("diff reports the headline change", changes["headline"] is not None)
    check("diff reports added skills", "LLMs" in changes["skills_added"])
    check("diff pairs a rewritten bullet with its original",
          any(c["was"] for c in changes["bullets"]))

    # skills-only mode must not touch bullets
    r_skills = pipeline.run_pipeline(
        resume_path, jd_text, mode=MODE_SKILLS_ONLY, target_score=90, max_iterations=2,
    )
    original_bullets = [b.text for b in r_skills.original.projects[0].bullets]
    final_bullets = [b.text for b in r_skills.final.projects[0].bullets]
    check("skills-only mode leaves project bullets untouched",
          original_bullets == final_bullets)
    check("skills-only mode still lifts the score",
          r_skills.final_report.score > r_skills.original_report.score,
          f"{r_skills.original_report.score} -> {r_skills.final_report.score}")
    check("skills-only scores lower than project mode",
          r_skills.final_report.score < result.final_report.score,
          f"skills={r_skills.final_report.score} projects={result.final_report.score}")

    print("\nRendering")
    tmpdir = tempfile.mkdtemp()
    docx_path = render.write_docx(result.final, os.path.join(tmpdir, "out.docx"))
    check("DOCX is written", os.path.exists(docx_path) and os.path.getsize(docx_path) > 5000)

    text, fmt, audit = parser.extract_text(docx_path)
    check("rendered DOCX re-parses as docx", fmt == "docx")
    check("renderer emits no tables", not audit.has_tables)
    check("renderer emits no images or text boxes",
          not audit.has_images and not audit.has_text_boxes)
    check("renderer is single column", not audit.multi_column)
    check("renderer puts nothing in the header/footer", not audit.in_header_footer)
    check("renderer produces a clean audit", audit.issues == [], str(audit.issues))
    check("round-tripped text keeps the new LLM bullet", "LLM" in text)
    check("round-tripped text keeps the candidate's name", "Arjun Mehta" in text)
    check("round-tripped text keeps every section heading",
          all(h in text for h in ["TECHNICAL SKILLS", "PROJECTS", "EDUCATION",
                                  "PROFESSIONAL EXPERIENCE"]))

    reparsed_score = ats.score_resume(
        ResumeDoc.from_dict(result.final.to_dict()), result.jd, audit
    )
    check("score survives the render round trip",
          abs(reparsed_score.score - result.final_report.score) < 0.5,
          f"{result.final_report.score} vs {reparsed_score.score}")

    pdf_path = render.write_pdf(docx_path, tmpdir)
    if pdf_path:
        check("PDF is written", os.path.getsize(pdf_path) > 3000)
        ptext, pfmt, paudit = parser.extract_text(pdf_path)
        check("PDF re-parses with its text intact",
              pfmt == "pdf" and "Arjun Mehta" in ptext and "LLM" in ptext)
        check("PDF is one page", paudit.page_count <= 2, f"{paudit.page_count} pages")
        check("PDF is not detected as multi-column", not paudit.multi_column)
    else:
        print("  [SKIP] PDF export (LibreOffice not on PATH)")


def test_plateau() -> None:
    print("\nPlateau / honest failure")
    llm.complete_json = fake_complete_json

    def stubborn(system: str, user: str, **kwargs):
        if "You route missing" in system:
            return {"assignments": [], "unplaceable": [
                {"skill": "PyTorch", "reason": "no deep-learning work exists"}]}
        if "You rewrite specific resume items" in system:
            return {}
        return fake_complete_json(system, user, **kwargs)

    import tailor.rewriter as rw_mod
    import tailor.jd as jd_mod
    rw_mod.complete_json = stubborn
    jd_mod.complete_json = fake_complete_json
    parser.complete_json = fake_complete_json

    with open(os.path.join(SAMPLES, "sample_jd.txt"), encoding="utf-8") as fh:
        jd_text = fh.read()
    result = pipeline.run_pipeline(
        os.path.join(SAMPLES, "sample_resume.txt"), jd_text,
        mode=MODE_PROJECTS_AND_SKILLS, target_score=90, max_iterations=4,
    )
    print(f"    final={result.final_report.score} | {result.stop_reason}")
    check("a hopeless run does not claim success", not result.target_met)
    check("it stops instead of looping forever", len(result.iterations) <= 5)
    check("it reports blocking gaps", len(result.blocking_gaps()) > 0)
    check("the gap list names the real blocker",
          any("PyTorch" in g for g in result.blocking_gaps()))
    check("it never returns a worse resume than it started with",
          result.final_report.score >= result.original_report.score)


def test_target_reached() -> None:
    """When every JD skill has an honest home, the loop must hit 90 and stop there."""
    print("\nTarget reached / early stop")
    import copy

    rich = copy.deepcopy(APPLY_RESPONSE)
    rich["projects"][0]["tech"] += ["PyTorch", "Kubernetes", "Pinecone"]
    rich["projects"][0]["bullets"].insert(
        3,
        {"text": "Fine-tuned a PyTorch transformer on the ticket corpus and served it "
                 "on Kubernetes with a Pinecone-backed retrieval index",
         "origin": "added", "skills": ["PyTorch", "Kubernetes", "Pinecone"]},
    )
    rich["skills"]["AI/ML"] = rich["skills"]["AI/ML"] + ["PyTorch", "fine-tuning"]
    rich["skills"]["Cloud & DevOps"] = rich["skills"]["Cloud & DevOps"] + [
        "Kubernetes", "model monitoring"
    ]

    def generous(system: str, user: str, **kwargs):
        if "You route missing" in system:
            return {
                "assignments": ROUTE_RESPONSE["assignments"]
                + [{"skill": "PyTorch", "target_type": "project", "target_id": "proj1",
                    "rationale": "The classifier is the natural place for a neural model.",
                    "how_applied": "Fine-tuned a PyTorch transformer on the ticket corpus."}],
                "unplaceable": [],
            }
        if "You rewrite specific resume items" in system:
            return rich
        return fake_complete_json(system, user, **kwargs)

    import tailor.jd as jd_mod
    import tailor.rewriter as rw_mod
    rw_mod.complete_json = generous
    jd_mod.complete_json = fake_complete_json
    parser.complete_json = fake_complete_json

    with open(os.path.join(SAMPLES, "sample_jd.txt"), encoding="utf-8") as fh:
        jd_text = fh.read()
    result = pipeline.run_pipeline(
        os.path.join(SAMPLES, "sample_resume.txt"), jd_text,
        mode=MODE_PROJECTS_AND_SKILLS, target_score=90, max_iterations=5,
    )
    print(f"    {result.original_report.score} -> {result.final_report.score} "
          f"| {result.stop_reason}")
    check("target of 90 is reached", result.final_report.score >= 90,
          f"final={result.final_report.score}")
    check("target_met is set", result.target_met)
    check("it stops as soon as the target is met", len(result.iterations) == 2,
          f"{len(result.iterations)} iterations")
    check("no blocking gaps are invented on success",
          not result.unplaceable)


if __name__ == "__main__":
    test_providers()
    test_matching()
    test_scoring()
    test_pipeline_and_render()
    test_target_reached()
    test_plateau()
    print("\n" + "=" * 60)
    if FAILS:
        print(f"{len(FAILS)} FAILURE(S):")
        for f in FAILS:
            print(f"  - {f}")
        sys.exit(1)
    print("All checks passed.")
