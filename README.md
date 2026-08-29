# ATS Resume Tailor

Rewrites a resume against a job description, then scores it with a built-in ATS
scorer and keeps rewriting until it clears 90% — or stops and tells you exactly
which real gap is blocking it.

```
resume (PDF/DOCX/TXT) ─┐
                       ├─→ parse ─→ score ─→ rewrite ─→ rescore ─→ … ─→ DOCX + PDF + report
job description ───────┘              ↑                    │
        scope dropdown ───────────────┴────────────────────┘
```

## The one thing that isn't as you asked

**No free ATS-checker API exists.** Jobscan, Resume Worded, Enhancv, LiveCareer,
Resume.io and the rest are web apps behind logins and paywalls; none of them
publish a scoring endpoint you can call. Scraping them would break the moment
they change their markup and violates their terms.

So `tailor/ats.py` reimplements the rubric those tools describe publicly. It is
free, unlimited, offline, reproducible, and — the part that actually matters —
it exposes *which* keyword in *which* section cost you points, which is what
lets the rewrite loop climb. A black-box score from a website could not steer
anything.

The scorer is deliberately stricter than most free web checkers in one respect:
a skill that appears only in your Skills list earns 70% credit, while one
demonstrated inside a project or job bullet earns 100%. That gap is the whole
reason "add it in projects and skills" beats "add in skills only".

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # pick a provider, paste its key
```

### Which model runs it — free options first

The pipeline is provider-agnostic. Everything except Anthropic speaks the
OpenAI chat-completions protocol, so switching is a base URL and a key.

| Provider | Cost | Weights | Get a key |
|---|---|---|---|
| **Groq** *(default)* | free | open — Llama 3.3 70B | [console.groq.com/keys](https://console.groq.com/keys) |
| **OpenRouter** | free on any `:free` model id | open — GLM, Nemotron, MiniMax | [openrouter.ai/keys](https://openrouter.ai/keys) |
| **Ollama** | free, **no key at all** | open, runs on your machine | [ollama.com/download](https://ollama.com/download) |
| Google Gemini | free tier | closed | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| Anthropic / OpenAI | paid | closed | — |

`python run_cli.py --list-providers` prints the same table with current defaults.

**Groq** is the recommended free start: open weights, no card, fastest inference.
Its free tier is roughly 30 requests/min and 1,000/day with a tight per-minute
token budget — one tailoring run costs 4–6 calls, so that's plenty, but the
per-minute token cap is the one you'll bump into on back-to-back runs. The client
backs off and retries automatically.

**Ollama** needs no key and has no quota — nothing leaves your machine:

```bash
ollama pull qwen2.5:14b-instruct     # ~9GB, comfortable on 16GB RAM
python run_cli.py --provider ollama --resume … --jd …
```

The tradeoff is honest: a 14B model is noticeably weaker at the skill-routing
judgement than a 70B one, and smaller models are the least reliable at the
structured JSON this pipeline needs. `tailor/llm.py` compensates — native JSON
mode, brace-matched extraction, `<think>` block stripping, and an explicit repair
round trip — but expect the occasional retry and slightly blunter bullets.

### Gotchas

`.env` is gitignored, so the repo ships only `.env.example` — you create the real
one. `load_dotenv()` resolves relative to where you launch from, so run everything
from the project root, and restart after editing `.env` since config is read once
at import. Or skip the file entirely: the Streamlit sidebar takes a provider and
key for the current session, held in memory only.

PDF export uses LibreOffice (`soffice` on PATH) when it is installed, because it
renders the real DOCX. Without it, `reportlab` draws the PDF directly — one
column, selectable text, hyphen bullets that survive text extraction. Either way
you get DOCX, PDF and TXT.

Free-tier quota on Gemini is **20 requests per day per model**, and one tailoring
run costs 4–6 of them. The bucket is per model, so switching the model in the
sidebar (`gemini-3.5-flash`, `gemini-3.1-flash-lite`, …) gets you a fresh
allowance. Reasoning models bill hidden thinking against `max_tokens`; `llm.py`
detects a truncated reply and retries with a doubled budget rather than failing.

## Run

```bash
streamlit run app.py                        # the UI with the dropdown

python run_cli.py --resume samples/sample_resume.txt \
                  --jd samples/sample_jd.txt \
                  --mode projects_and_skills --provider groq --json

python run_cli.py --list-providers          # providers, defaults, key URLs
python tests/test_offline.py                # 71 checks, no API key needed
```

The ATS score is computed locally and never calls a model, so it costs nothing
and is byte-identical on every run regardless of which provider you pick. The
model is used only for parsing and rewriting.

## The three scopes

| Dropdown option | What changes | Typical ceiling |
|---|---|---|
| **Add in skills only** | Skills section only | ~80% — keywords with no evidence behind them |
| **Add it in projects and skill** | The right project is rewritten to demonstrate the skill, then the skill is mirrored into Skills | ~90% |
| **Add overall in the resume** | Headline, summary, skills, projects and work bullets | highest |

## How a skill gets routed into a project

This is the part you asked about specifically. When the JD demands `LLMs` and
your resume never says it, the router (`tailor/rewriter.py`) is handed every
project with its real tech stack and bullets, and picks the one where that skill
would credibly have been used — judged by technical adjacency, not name
similarity.

On the sample resume it produces:

```
LLMs              -> Support Ticket Classifier
    "It is already a text-classification/NLP system, so an LLM is a
     credible evolution."
    → "Built an LLM-based intent classifier that routes incoming support
       tickets into 12 categories with 87% accuracy, replacing a TF-IDF baseline"

RAG pipelines     -> Support Ticket Classifier
vector database   -> Support Ticket Classifier
AWS               -> Software Engineer @ Cobalt Systems
```

The e-commerce project is the larger, flashier one — and the router correctly
leaves it alone, because an LLM has no business there.

Skills with no honest home come back as **unplaceable** rather than being bolted
onto an unrelated project:

```
PyTorch     — all modelling work used scikit-learn; no deep-learning project exists
Kubernetes  — nothing on the resume was orchestrated beyond Docker Compose
```

## The 90% rule

The loop runs up to 5 rounds. Each round feeds the previous round's gap report
back into the rewriter. It stops when the target is met, when the score
plateaus (< 1 point of movement), or when rounds run out — and it always returns
the best-scoring version, never a worse one than you uploaded.

When it lands short, it says why. On the sample resume it stops at **89.6** and
names PyTorch as the blocker: the JD requires it and nothing in that candidate's
history supports the claim. Padding the number past 90 there would mean writing
a sentence you cannot defend in an interview, so it doesn't.

**Every added or rewritten bullet is tagged and surfaced in a review panel
before you download anything.** Read them. They are claims about you.

## Editing the result

The **Edit & preview** tab makes every field editable in place — headline,
summary, each skill category, and every bullet as one line of text. The preview,
the ATS score and the DOCX/PDF/TXT downloads all update as you type.

Editing and re-scoring are entirely local: no model call, no quota. Tweaking a
word keeps that bullet's provenance, so the "what changed" review stays accurate
after you edit.

## Scoring rubric

| Component | Weight | What it measures |
|---|---|---|
| Hard skills | 40 | Required skills ×2, preferred ×1; weighted by *where* each appears |
| Keyword coverage | 20 | Verbatim JD keyword surface forms found anywhere |
| Title match | 10 | Target job title in the headline, summary or a role title |
| Responsibilities | 5 | Partial-credit token overlap between JD duties and your bullets |
| Parseability | 15 | Tables, images, text boxes, columns, header/footer content, fonts, length |
| Completeness | 10 | Email, phone, location, links, summary, education, skills |

Matching handles acronym aliases (`LLM` ↔ `Large Language Models`, `k8s` ↔
`Kubernetes`, ~40 groups), plural/singular, hyphen and slash splitting
(`LLM-powered` matches `LLM`), and fuzzy near-misses at 88% similarity.

## Layout

```
app.py              Streamlit UI
run_cli.py          CLI
tailor/
  config.py         modes, rubric weights, target, iteration caps
  schema.py         ResumeDoc / JobDescription, bullet provenance
  providers.py      Groq / OpenRouter / Ollama / Gemini / Anthropic / OpenAI
  llm.py            provider-agnostic JSON completion, repair + backoff
  parser.py         PDF/DOCX extraction + format audit
  jd.py             JD -> structured requirements
  ats.py            the scorer (no network, no vendor API)
  rewriter.py       skill->project routing + the three rewrite scopes
  pipeline.py       the score/rewrite/rescore loop
  render.py         ATS-safe DOCX, PDF, plain text
tests/test_offline.py   full pipeline with a stubbed model
samples/                example resume, JD, and generated output
```

## Honesty constraints

Baked into every rewrite prompt and enforced by the patch applier: no invented
employers, titles, degrees, certifications, dates or metrics. Numbers must
already exist in the resume. Only framing changes — the domain, subject matter
and outcome of each project stay what they actually were. Pre-existing skills
are never silently dropped.
