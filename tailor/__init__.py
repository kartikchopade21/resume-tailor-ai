"""JD-aware resume tailoring pipeline with a built-in ATS scorer."""
from .config import (
    MODE_FULL_RESUME,
    MODE_LABELS,
    MODE_HELP,
    MODE_PROJECTS_AND_SKILLS,
    MODE_SKILLS_ONLY,
    SETTINGS,
)
from . import providers
from .ats import ATSReport, score_resume
from .jd import parse_jd
from .parser import load_resume, load_resume_bytes
from .pipeline import PipelineResult, diff_summary, run_pipeline
from .render import to_plain_text, write_docx, write_pdf
from .schema import JobDescription, ResumeDoc

__all__ = [
    "ATSReport", "JobDescription", "PipelineResult", "ResumeDoc", "SETTINGS",
    "MODE_SKILLS_ONLY", "MODE_PROJECTS_AND_SKILLS", "MODE_FULL_RESUME",
    "MODE_LABELS", "MODE_HELP",
    "providers",
    "diff_summary", "load_resume", "load_resume_bytes", "parse_jd",
    "run_pipeline", "score_resume", "to_plain_text", "write_docx", "write_pdf",
]
__version__ = "1.0.0"
