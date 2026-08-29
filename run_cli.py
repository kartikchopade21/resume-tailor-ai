#!/usr/bin/env python3
"""CLI entry point.

    python run_cli.py --resume resume.pdf --jd jd.txt --mode projects_and_skills
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from tailor import (
    MODE_FULL_RESUME,
    MODE_LABELS,
    MODE_PROJECTS_AND_SKILLS,
    MODE_SKILLS_ONLY,
    SETTINGS,
    diff_summary,
    providers,
    run_pipeline,
    to_plain_text,
    write_docx,
    write_pdf,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Tailor a resume to a job description and score it.")
    ap.add_argument("--resume", help="path to resume (.pdf/.docx/.txt)")
    ap.add_argument("--jd", help="path to a job description text file")
    ap.add_argument(
        "--mode", default=MODE_PROJECTS_AND_SKILLS,
        choices=[MODE_SKILLS_ONLY, MODE_PROJECTS_AND_SKILLS, MODE_FULL_RESUME],
        help="; ".join(f"{k}={v}" for k, v in MODE_LABELS.items()),
    )
    ap.add_argument("--target", type=float, default=SETTINGS.target_score)
    ap.add_argument("--max-iterations", type=int, default=SETTINGS.max_iterations)
    ap.add_argument("--out", default="out", help="output directory")
    ap.add_argument("--json", action="store_true", help="also dump the full report as JSON")
    ap.add_argument(
        "--provider", default=None, choices=list(providers.PROVIDERS),
        help="model provider (default: whichever key is set in .env)",
    )
    ap.add_argument("--model", default=None, help="override the provider's default model")
    ap.add_argument(
        "--list-providers", action="store_true", help="show providers and exit",
    )
    args = ap.parse_args()

    if args.list_providers:
        for p in providers.PROVIDERS.values():
            tag = "FREE" if p.free else "paid"
            print(f"{p.key:<12} [{tag}]  default model: {p.default_model}")
            print(f"{'':14}{p.notes}")
            print(f"{'':14}key: {p.console_url}\n")
        return 0

    if not args.resume or not args.jd:
        ap.error("--resume and --jd are required (except with --list-providers)")

    if args.provider or args.model:
        SETTINGS.use_provider(args.provider or SETTINGS.provider, model=args.model)

    provider = providers.get(SETTINGS.provider)
    if not SETTINGS.configured:
        print(f"error: {provider.label} needs a key — set {provider.env_var} in .env "
              f"(get one at {provider.console_url}), or use --provider ollama to run "
              f"locally with no key", file=sys.stderr)
        return 2
    print(f"  provider: {provider.key} · model: {SETTINGS.model}", file=sys.stderr)

    with open(args.jd, "r", encoding="utf-8") as fh:
        jd_text = fh.read()

    result = run_pipeline(
        args.resume, jd_text, mode=args.mode,
        target_score=args.target, max_iterations=args.max_iterations,
        progress=lambda msg: print(f"  {msg}", file=sys.stderr),
    )

    os.makedirs(args.out, exist_ok=True)
    base = (result.final.contact.name or "resume").replace(" ", "_")
    docx_path = write_docx(result.final, os.path.join(args.out, f"{base}_tailored.docx"))
    pdf_path = write_pdf(docx_path, args.out, result.final)
    txt_path = os.path.join(args.out, f"{base}_tailored.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(to_plain_text(result.final))

    print()
    print(f"ATS score : {result.original_report.score}  ->  {result.final_report.score} "
          f"(target {args.target})")
    print(f"Status    : {'TARGET MET' if result.target_met else 'BELOW TARGET'}")
    print(f"Reason    : {result.stop_reason}")

    if result.assignments:
        print("\nSkill routing:")
        for a in result.assignments:
            print(f"  {a['skill']:<28} -> {a['target']}")
            print(f"      {a['rationale']}")

    if not result.target_met:
        print("\nBlocking gaps:")
        for gap in result.blocking_gaps():
            print(f"  - {gap}")

    changes = diff_summary(result.original, result.final)
    if changes["bullets"]:
        print(f"\n{len(changes['bullets'])} bullet(s) written or rewritten — review before sending.")

    print(f"\nWrote: {docx_path}")
    if pdf_path:
        print(f"Wrote: {pdf_path}")
    print(f"Wrote: {txt_path}")

    if args.json:
        report_path = os.path.join(args.out, "ats_report.json")
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "original_score": result.original_report.score,
                    "final_score": result.final_report.score,
                    "target_met": result.target_met,
                    "stop_reason": result.stop_reason,
                    "mode": result.mode,
                    "assignments": result.assignments,
                    "unplaceable": result.unplaceable,
                    "final_report": result.final_report.to_dict(),
                    "resume": result.final.to_dict(),
                },
                fh, indent=2,
            )
        print(f"Wrote: {report_path}")
    return 0 if result.target_met else 1


if __name__ == "__main__":
    raise SystemExit(main())
