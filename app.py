"""Streamlit UI:  resume input -> JD input -> mode dropdown -> tailored resume + ATS score."""
from __future__ import annotations

import os
import tempfile
import uuid

import streamlit as st

# Hosted deployments have no .env — Streamlit Community Cloud supplies keys via
# the app's Secrets panel instead. Copy them into the environment before tailor
# is imported, since config.py reads os.environ once at import time.
# setdefault, so a real .env still wins when running locally.
try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, str):
            os.environ.setdefault(_key, _value)
except Exception:      # no secrets configured at all — normal for local runs
    pass

from tailor import llm, providers  # noqa: E402
from tailor import (  # noqa: E402
    MODE_FULL_RESUME,
    MODE_HELP,
    MODE_LABELS,
    MODE_PROJECTS_AND_SKILLS,
    MODE_SKILLS_ONLY,
    SETTINGS,
    diff_summary,
    run_pipeline,
    score_resume,
    to_plain_text,
    write_docx,
    write_pdf,
)
from tailor.schema import Bullet  # noqa: E402

st.set_page_config(page_title="ATS Resume Tailor", page_icon="🎯", layout="wide")

MODE_ORDER = [MODE_SKILLS_ONLY, MODE_PROJECTS_AND_SKILLS, MODE_FULL_RESUME]


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def _csv(text: str) -> list[str]:
    return [p.strip() for p in (text or "").split(",") if p.strip()]


def _rebuild_bullets(texts: list[str], previous: list[Bullet]) -> list[Bullet]:
    """Keep each bullet's provenance when the user only tweaks its wording."""
    out: list[Bullet] = []
    for i, text in enumerate(texts):
        if i < len(previous):
            old = previous[i]
            out.append(Bullet(text=text, origin=old.origin, skills=list(old.skills)))
        else:
            out.append(Bullet(text=text, origin="added"))
    return out


def render_editor(doc, token: str):
    """Editable copy of the tailored resume. Returns the edited ResumeDoc.

    Everything here is local — editing and re-scoring never call the model, so
    fixing a word costs nothing and takes no quota.
    """
    edited = doc.copy()
    k = lambda name: f"{token}:{name}"  # noqa: E731 — keys must change per run

    with st.expander("Header and summary", expanded=True):
        c1, c2 = st.columns(2)
        edited.contact.name = c1.text_input("Name", doc.contact.name, key=k("name"))
        edited.contact.headline = c2.text_input(
            "Headline", doc.contact.headline, key=k("headline"))
        c3, c4, c5 = st.columns(3)
        edited.contact.location = c3.text_input(
            "Location", doc.contact.location, key=k("loc"))
        edited.contact.phone = c4.text_input("Phone", doc.contact.phone, key=k("phone"))
        edited.contact.email = c5.text_input("Email", doc.contact.email, key=k("email"))
        c6, c7 = st.columns(2)
        edited.contact.linkedin = c6.text_input(
            "LinkedIn", doc.contact.linkedin, key=k("li"))
        edited.contact.github = c7.text_input("GitHub", doc.contact.github, key=k("gh"))
        edited.summary = st.text_area(
            "Professional summary", doc.summary, height=120, key=k("summary"))

    with st.expander("Skills", expanded=True):
        st.caption("One category per row, skills separated by commas.")
        new_skills: dict[str, list[str]] = {}
        for idx, (cat, vals) in enumerate(doc.skills.items()):
            c1, c2 = st.columns([1, 3])
            name = c1.text_input(
                "Category", cat, key=k(f"skcat{idx}"), label_visibility="collapsed")
            body = c2.text_input(
                "Skills", ", ".join(vals), key=k(f"skval{idx}"),
                label_visibility="collapsed")
            if name.strip() and _csv(body):
                new_skills[name.strip()] = _csv(body)
        c1, c2 = st.columns([1, 3])
        extra_cat = c1.text_input(
            "New category", "", key=k("skcatnew"), placeholder="New category",
            label_visibility="collapsed")
        extra_val = c2.text_input(
            "New skills", "", key=k("skvalnew"), placeholder="skill, skill, skill",
            label_visibility="collapsed")
        if extra_cat.strip() and _csv(extra_val):
            new_skills[extra_cat.strip()] = _csv(extra_val)
        edited.skills = new_skills

    if doc.experience:
        with st.expander("Experience", expanded=True):
            for idx, exp in enumerate(doc.experience):
                c1, c2, c3 = st.columns([2, 2, 1])
                edited.experience[idx].title = c1.text_input(
                    "Title", exp.title, key=k(f"exp{idx}t"))
                edited.experience[idx].company = c2.text_input(
                    "Company", exp.company, key=k(f"exp{idx}c"))
                edited.experience[idx].dates = c3.text_input(
                    "Dates", exp.dates, key=k(f"exp{idx}d"))
                text = st.text_area(
                    "Bullets — one per line", "\n".join(b.text for b in exp.bullets),
                    height=max(90, 26 * len(exp.bullets)), key=k(f"exp{idx}b"))
                edited.experience[idx].bullets = _rebuild_bullets(
                    _lines(text), exp.bullets)
                st.divider()

    if doc.projects:
        with st.expander("Projects", expanded=True):
            for idx, proj in enumerate(doc.projects):
                c1, c2 = st.columns([2, 3])
                edited.projects[idx].name = c1.text_input(
                    "Project", proj.name, key=k(f"pr{idx}n"))
                edited.projects[idx].subtitle = c2.text_input(
                    "Context", proj.subtitle, key=k(f"pr{idx}s"))
                edited.projects[idx].tech = _csv(st.text_input(
                    "Technologies — comma separated", ", ".join(proj.tech),
                    key=k(f"pr{idx}t")))
                text = st.text_area(
                    "Bullets — one per line", "\n".join(b.text for b in proj.bullets),
                    height=max(90, 26 * len(proj.bullets)), key=k(f"pr{idx}b"))
                edited.projects[idx].bullets = _rebuild_bullets(
                    _lines(text), proj.bullets)
                st.divider()

    if doc.education or doc.certifications or doc.extras:
        with st.expander("Education, certifications and other sections"):
            for idx, ed in enumerate(doc.education):
                c1, c2, c3 = st.columns([2, 2, 1])
                edited.education[idx].degree = c1.text_input(
                    "Degree", ed.degree, key=k(f"ed{idx}d"))
                edited.education[idx].institution = c2.text_input(
                    "Institution", ed.institution, key=k(f"ed{idx}i"))
                edited.education[idx].dates = c3.text_input(
                    "Year", ed.dates, key=k(f"ed{idx}y"))
                edited.education[idx].details = st.text_input(
                    "Details", ed.details, key=k(f"ed{idx}x"))
            if doc.certifications:
                edited.certifications = _lines(st.text_area(
                    "Certifications — one per line", "\n".join(doc.certifications),
                    key=k("certs")))
            for idx, (name, vals) in enumerate(doc.extras.items()):
                edited.extras[name] = _lines(st.text_area(
                    f"{name} — one per line", "\n".join(vals), key=k(f"ex{idx}"),
                    height=max(90, 26 * len(vals))))

    return edited


def offer_downloads(doc, key_prefix: str) -> None:
    """DOCX / PDF / TXT of whatever the editor currently holds."""
    tmpdir = tempfile.mkdtemp()
    base = (doc.contact.name or "resume").replace(" ", "_")
    docx_path = write_docx(doc, os.path.join(tmpdir, f"{base}_tailored.docx"))
    pdf_path = write_pdf(docx_path, tmpdir, doc)

    cols = st.columns(3)
    with open(docx_path, "rb") as fh:
        cols[0].download_button(
            "⬇️ DOCX", fh.read(), file_name=os.path.basename(docx_path),
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True, key=f"{key_prefix}docx")
    if pdf_path:
        with open(pdf_path, "rb") as fh:
            cols[1].download_button(
                "⬇️ PDF", fh.read(), file_name=os.path.basename(pdf_path),
                mime="application/pdf", use_container_width=True,
                key=f"{key_prefix}pdf")
    else:
        cols[1].caption("PDF needs LibreOffice or `pip install reportlab`.")
    cols[2].download_button(
        "⬇️ Text", to_plain_text(doc), file_name=f"{base}_tailored.txt",
        mime="text/plain", use_container_width=True, key=f"{key_prefix}txt")


def score_color(score: float, target: float) -> str:
    if score >= target:
        return "#16794c"
    if score >= target - 10:
        return "#b06d00"
    return "#b42318"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Model")

    names = list(providers.PROVIDERS)
    chosen = st.selectbox(
        "Provider", names,
        index=names.index(SETTINGS.provider) if SETTINGS.provider in names else 0,
        format_func=lambda n: providers.PROVIDERS[n].label,
    )
    if chosen != SETTINGS.provider:
        SETTINGS.use_provider(chosen)

    provider = providers.get(SETTINGS.provider)
    st.caption(provider.notes)

    model = st.text_input("Model", value=SETTINGS.model)
    if model.strip() and model.strip() != SETTINGS.model:
        SETTINGS.model = model.strip()
        llm.reset_client()

    if not provider.needs_key:
        st.success("No API key needed — this runs on your machine.")
        st.caption(f"Endpoint: `{provider.base_url}`")
    elif provider.key_from_env():
        st.caption(f"API key: ✅ loaded from `{provider.env_var}`")
    else:
        # Held in memory for this session only — never written to disk or logged.
        typed = st.text_input(
            "API key", type="password",
            help="Used for this session only, not saved to disk. To persist it, "
                 f"set {provider.env_var} in .env.",
        )
        if typed.strip():
            SETTINGS.api_key = typed.strip()
            llm.reset_client()
            st.caption("API key: ✅ set for this session")
        else:
            SETTINGS.api_key = ""
            st.caption("API key: ❌ needed")
            st.markdown(
                f"[Get a {'free ' if provider.free else ''}key →]({provider.console_url})"
            )

    st.divider()
    st.header("Settings")
    target = st.slider("Target ATS score", 70, 100, int(SETTINGS.target_score), 1)
    max_iters = st.slider("Max rewrite rounds", 1, 8, int(SETTINGS.max_iterations), 1)
    st.divider()
    st.caption(
        "The ATS score is computed locally — it never calls a model or a vendor "
        "API, so the number costs nothing and is identical on every run. "
        "The model is used only to parse and rewrite."
    )

st.title("🎯 ATS Resume Tailor")
st.caption("Rewrites your resume against a job description, then scores it until it clears the bar.")

# ------------------------------------------------------------------ inputs
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("1 · Resume")
    uploaded = st.file_uploader(
        "PDF, DOCX or TXT", type=["pdf", "docx", "txt"], label_visibility="collapsed"
    )
    if uploaded:
        st.success(f"{uploaded.name} · {uploaded.size // 1024} KB")

with col_b:
    st.subheader("2 · Job description")
    jd_text = st.text_area(
        "Paste the full JD", height=260, label_visibility="collapsed",
        placeholder="Paste the complete job description here — responsibilities, "
                    "requirements, preferred qualifications…",
    )

st.subheader("3 · What should be changed")
mode = st.selectbox(
    "Scope",
    MODE_ORDER,
    index=1,
    format_func=lambda m: MODE_LABELS[m],
    label_visibility="collapsed",
)
st.caption(MODE_HELP[mode])

run = st.button("Tailor my resume", type="primary", use_container_width=True,
                disabled=not (uploaded and jd_text.strip() and SETTINGS.configured))

if not SETTINGS.configured:
    st.warning(
        f"**{provider.label}** needs an API key. "
        f"[Get one {'free' if provider.free else ''} here]({provider.console_url}) and "
        f"paste it in the sidebar — or switch the provider to **Ollama**, which runs "
        f"locally with no key at all."
    )

# ------------------------------------------------------------------- run
if run:
    status = st.status("Starting…", expanded=True)
    try:
        result = run_pipeline(
            (uploaded.getvalue(), uploaded.name),
            jd_text,
            mode=mode,
            target_score=float(target),
            max_iterations=int(max_iters),
            progress=lambda msg: status.update(label=msg),
        )
        status.update(label="Done", state="complete", expanded=False)
        st.session_state["result"] = result
        # new run => new widget keys, so the editor reloads instead of showing
        # the previous resume's edits
        st.session_state["token"] = uuid.uuid4().hex[:8]
    except Exception as exc:
        status.update(label="Failed", state="error")
        st.error(str(exc))

# ---------------------------------------------------------------- results
result = st.session_state.get("result")
if result:
    before = result.original_report.score
    after = result.final_report.score
    tgt = float(target)

    m1, m2, m3 = st.columns(3)
    m1.metric("Original ATS score", f"{before}%")
    m2.metric("Tailored ATS score", f"{after}%", delta=f"{result.delta:+.1f}")
    m3.metric("Target", f"{tgt:.0f}%",
              delta="met" if result.target_met else "not met",
              delta_color="normal" if result.target_met else "inverse")

    st.markdown(
        f"<div style='height:14px;border-radius:7px;background:#eceff3;overflow:hidden'>"
        f"<div style='height:100%;width:{min(after, 100)}%;"
        f"background:{score_color(after, tgt)}'></div></div>",
        unsafe_allow_html=True,
    )
    st.caption(result.stop_reason)

    if not result.target_met:
        gaps = result.blocking_gaps()
        if gaps:
            st.warning(
                "**Could not reach the target honestly. What is blocking it:**\n\n"
                + "\n".join(f"- {g}" for g in gaps)
            )

    tabs = st.tabs(
        ["Score breakdown", "What changed", "Skill → project routing",
         "Edit & preview", "Download"]
    )

    # --- breakdown
    with tabs[0]:
        r = result.final_report
        rows = []
        for label, value in r.breakdown.__dict__.items():
            cap = r.max_breakdown.get(label, 0)
            rows.append(
                {
                    "Component": label.replace("_", " ").title(),
                    "Score": round(value, 1),
                    "Max": cap,
                    "%": f"{(value / cap * 100):.0f}%" if cap else "—",
                }
            )
        st.dataframe(rows, hide_index=True, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Matched keywords**")
            st.write(", ".join(r.matched_keywords) or "—")
        with c2:
            st.markdown("**Still missing**")
            st.write(", ".join(r.missing_keywords) or "—")
        if r.missing_required:
            st.error("Required skills still absent: " + ", ".join(r.missing_required))
        if r.weak_skills:
            st.info(
                "Listed in Skills but never demonstrated in a project or role: "
                + ", ".join(r.weak_skills)
            )
        with st.expander("Score history"):
            st.dataframe(
                [{"Round": it.index, "Score": it.score} for it in result.iterations],
                hide_index=True, use_container_width=True,
            )

    # --- diff
    with tabs[1]:
        changes = diff_summary(result.original, result.final)
        if changes["headline"]:
            st.markdown("**Headline**")
            st.caption(f"was: {changes['headline'][0] or '(none)'}")
            st.write(changes["headline"][1])
        if changes["summary"]:
            st.markdown("**Summary**")
            st.caption(f"was: {changes['summary'][0] or '(none)'}")
            st.write(changes["summary"][1])
        if changes["skills_added"]:
            st.markdown("**Skills added**")
            st.write(", ".join(changes["skills_added"]))
        st.markdown("**Bullets written or rewritten**")
        if not changes["bullets"]:
            st.caption("No bullet-level changes in this mode.")
        for ch in changes["bullets"]:
            tag = "🆕 new" if ch["origin"] == "added" else "✏️ rewritten"
            with st.container(border=True):
                st.caption(f"{tag} · {ch['where']}"
                           + (f" · demonstrates: {', '.join(ch['skills'])}" if ch["skills"] else ""))
                if ch["was"]:
                    st.markdown(f"<span style='color:#b42318'>was: {ch['was']}</span>",
                                unsafe_allow_html=True)
                st.write(ch["text"])
        st.info(
            "Every line above is a claim you will have to defend in an interview. "
            "Read them before you send this resume anywhere."
        )

    # --- routing
    with tabs[2]:
        st.caption(
            "For each skill the JD wanted and the resume lacked, which project or role "
            "the pipeline judged the most plausible home — and why."
        )
        if result.assignments:
            for a in result.assignments:
                with st.container(border=True):
                    st.markdown(f"**{a['skill']}** → {a['target']} ({a['target_type']})")
                    st.caption(a["rationale"])
                    st.write(a["how_applied"])
        else:
            st.caption("No routing was needed in this mode.")
        if result.unplaceable:
            st.warning("**Skills with no honest home on this resume**")
            for item in result.unplaceable:
                st.markdown(f"- **{item['skill']}** — {item['reason']}")

    # --- edit + preview
    with tabs[3]:
        token = st.session_state.setdefault("token", "v0")
        st.caption(
            "Edit any field below and the preview, the ATS score and the downloads "
            "update as you type. This is all local — no model call, no quota used."
        )
        if st.button("↺ Reset to the tailored version", key="reset_edits"):
            for key in [x for x in st.session_state if x.startswith(f"{token}:")]:
                del st.session_state[key]
            st.rerun()

        edited = render_editor(result.final, token)
        st.session_state["edited"] = edited

        edited_report = score_resume(edited, result.jd)
        e1, e2, e3 = st.columns(3)
        e1.metric("Tailored ATS score", f"{after}%")
        e2.metric("Your edited version", f"{edited_report.score}%",
                  delta=f"{edited_report.score - after:+.1f}")
        e3.metric("Target", f"{tgt:.0f}%",
                  delta="met" if edited_report.score >= tgt else "not met",
                  delta_color="normal" if edited_report.score >= tgt else "inverse")
        if edited_report.missing_required:
            st.warning("Required skills not in your edit: "
                       + ", ".join(edited_report.missing_required))

        st.markdown("**Download what you see**")
        offer_downloads(edited, "edit_")

        with st.expander("Plain-text preview", expanded=True):
            st.code(to_plain_text(edited), language=None)

    # --- download
    with tabs[4]:
        # the editor tab renders first, so its edits are already in session state
        final_doc = st.session_state.get("edited") or result.final
        if final_doc is not result.final:
            st.caption("Includes your edits from the **Edit & preview** tab.")
        st.markdown("**Tailored resume** — ATS-safe: one column, real text, no tables.")
        offer_downloads(final_doc, "dl_")
