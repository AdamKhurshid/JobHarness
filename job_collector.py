#!/usr/bin/env python3
"""
arbeitnow_qwen.py

Search the Arbeitnow job board API for jobs matching a text query, then enrich
each job with structured fields (qualifications, yearly salary, hourly wage)
extracted from the raw HTML description by a local Qwen3 8B model running in
Ollama.

Usage
-----
    python arbeitnow_qwen.py "python django"
    python arbeitnow_qwen.py "react" --limit 3 --model qwen3:8b
    python arbeitnow_qwen.py "golang remote" --output ./jobs.json

Requirements
------------
    pip install requests
    Ollama running locally (https://ollama.com)
    ollama pull qwen3:8b

Output
------
A JSON file (default: ./download/jobs_<query>_<timestamp>.json) containing the
original Arbeitnow fields minus `description`, plus:

    qualifications : list[str]   - atomic list of qualifications/requirements
    yearlySalary   : str | "NaN" - yearly salary range string, or "NaN"
    hourlyWage     : str | "NaN" - hourly wage string, or "NaN"
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from html.parser import HTMLParser


ARBEITNOW_API = "https://www.arbeitnow.com/api/job-board-api"
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_LIMIT = 5
HTTP_TIMEOUT = 30


# --------------------------------------------------------------------------- #
# HTML stripping (stdlib only - no bs4 dependency)
# --------------------------------------------------------------------------- #
class _Stripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:  # noqa: D401
        self._chunks.append(data)

    def get_text(self) -> str:
        text = "".join(self._chunks)
        text = html.unescape(text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text


_BLOCK_TAGS = (
    "p", "div", "br", "li", "ul", "ol",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "tr", "table", "section", "article", "header", "footer",
)


def strip_html(raw: str) -> str:
    """Convert an HTML job description to plain, readable text."""
    if not raw:
        return ""

    # Insert a separator after closing block-level tags so list items and
    # headings don't run together (e.g. "</li><li>" -> "</li> \n<li>").
    block_close_re = re.compile(
        r"(</(?:" + "|".join(_BLOCK_TAGS) + r")>)",
        flags=re.IGNORECASE,
    )
    spaced = block_close_re.sub(r"\1 \n", raw)

    stripper = _Stripper()
    try:
        stripper.feed(spaced)
    except Exception:
        # Fallback: regex strip if the parser chokes
        text = re.sub(r"<[^>]+>", " ", raw)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    return stripper.get_text()


# --------------------------------------------------------------------------- #
# Arbeitnow
# --------------------------------------------------------------------------- #
def fetch_jobs(query: str, limit: int) -> list[dict[str, Any]]:
    """Fetch up to `limit` jobs from Arbeitnow matching `query`."""
    params = {"search": query}
    print(f"[arbeitnow] GET {ARBEITNOW_API}?search={query}")
    resp = requests.get(ARBEITNOW_API, params=params, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()

    jobs = payload.get("data", []) or []
    print(f"[arbeitnow] received {len(jobs)} job(s); using first {limit}")
    return jobs[:limit]


# --------------------------------------------------------------------------- #
# Ollama / Qwen3 8B
# --------------------------------------------------------------------------- #
PROMPT_TEMPLATE = """You are a precise job-description parser.

Read the job description below and extract exactly three fields:

1. "qualifications" - an ATOMIC list of every distinct qualification, \
requirement, must-have skill, or nice-to-have mentioned. Each list item must \
be ONE short, self-contained statement (e.g. "3+ years of Python / Django", \
"Experience building REST APIs"). Do not merge multiple requirements into a \
single item. Preserve currency symbols, units, and numbers verbatim. \
If none are mentioned, return an empty list [].

2. "yearlySalary" - the yearly salary range or single salary as a string \
(e.g. "€70,000 - €85,000", "$120k", "£50,000"). If only an hourly or monthly \
figure is given, leave this as "NaN".

3. "hourlyWage" - the hourly wage as a string (e.g. "$25/hr", "€18/hour"). \
If only a yearly salary is given, leave this as "NaN".

Return STRICT JSON only - no prose, no markdown fences, no comments. The JSON \
object must have exactly these three keys: "qualifications", "yearlySalary", \
"hourlyWage".

Job description:
----------------
{description}
----------------
"""


def analyze_description(description: str, model: str) -> dict[str, Any]:
    """Call Ollama with Qwen3 8B to extract structured fields from a description."""
    clean = strip_html(description)
    if not clean:
        return {"qualifications": [], "yearlySalary": "NaN", "hourlyWage": "NaN"}

    prompt = PROMPT_TEMPLATE.format(description=clean)

    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",     # force JSON output
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 1024,
        },
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    raw_text = data.get("response", "").strip()

    # Ollama in JSON mode should return a JSON string, but be defensive.
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to extract the first {...} block
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if not match:
            print(f"[qwen] WARNING: could not parse model output as JSON:\n{raw_text[:300]}",
                  file=sys.stderr)
            return {"qualifications": [], "yearlySalary": "NaN", "hourlyWage": "NaN"}
        parsed = json.loads(match.group(0))

    # Normalize / validate fields
    quals = parsed.get("qualifications")
    if not isinstance(quals, list):
        quals = [str(quals)] if quals else []
    quals = [str(q).strip() for q in quals if str(q).strip()]

    yearly = parsed.get("yearlySalary")
    if yearly in (None, "", "null"):
        yearly = "NaN"
    else:
        yearly = str(yearly).strip()

    hourly = parsed.get("hourlyWage")
    if hourly in (None, "", "null"):
        hourly = "NaN"
    else:
        hourly = str(hourly).strip()

    return {
        "qualifications": quals,
        "yearlySalary": yearly,
        "hourlyWage": hourly,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Search Arbeitnow jobs and enrich descriptions with Qwen3 8B via Ollama."
    )
    p.add_argument("query", help="Text to search the Arbeitnow job board for.")
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                   help=f"Max jobs to fetch and analyze (default: {DEFAULT_LIMIT}).")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Ollama model name (default: {DEFAULT_MODEL}).")
    p.add_argument("--output", default=None,
                   help="Output JSON file path (default: ./download/jobs_<query>_<ts>.json).")
    p.add_argument("--ollama-url", default=OLLAMA_URL,
                   help=f"Ollama API endpoint (default: {OLLAMA_URL}).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    global OLLAMA_URL
    OLLAMA_URL = args.ollama_url

    # 1) Fetch
    try:
        jobs = fetch_jobs(args.query, args.limit)
    except requests.RequestException as e:
        print(f"[error] failed to fetch jobs from Arbeitnow: {e}", file=sys.stderr)
        return 1

    if not jobs:
        print(f"[info] no jobs found for query '{args.query}'.")
        return 0

    # 2) Enrich each job
    enriched: list[dict[str, Any]] = []
    for i, job in enumerate(jobs, start=1):
        slug = job.get("slug", f"<job-{i}>")
        print(f"[qwen] ({i}/{len(jobs)}) analyzing: {slug}")

        description = job.pop("description", "")
        try:
            extracted = analyze_description(description, args.model)
        except requests.RequestException as e:
            print(f"[error] Ollama call failed for {slug}: {e}", file=sys.stderr)
            print("[hint] is Ollama running? try: ollama serve && ollama pull "
                  f"{args.model}", file=sys.stderr)
            return 2
        except Exception as e:
            print(f"[error] unexpected error analyzing {slug}: {e}", file=sys.stderr)
            extracted = {"qualifications": [], "yearlySalary": "NaN", "hourlyWage": "NaN"}

        # Build the final record - keep original fields, drop description, add new
        record = dict(job)
        record["qualifications"] = extracted["qualifications"]
        record["yearlySalary"] = extracted["yearlySalary"]
        record["hourlyWage"] = extracted["hourlyWage"]
        enriched.append(record)

        # Be polite to the local model
        time.sleep(0.2)

    # 3) Persist
    if args.output:
        out_path = Path(args.output)
    else:
        download_dir = Path("/home/abigdam/codingProjects/JobHarness/data")
        download_dir.mkdir(parents=True, exist_ok=True)
        safe_query = re.sub(r"[^a-z0-9]+", "_", args.query.lower()).strip("_") or "query"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = download_dir / f"jobs_{safe_query}_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[done] saved {len(enriched)} enriched job(s) to: {out_path}")

    # Also print a quick summary to the terminal
    for r in enriched:
        print(f"  - {r.get('title')} @ {r.get('company_name')} | "
              f"salary={r.get('yearlySalary')} wage={r.get('hourlyWage')} | "
              f"quals={len(r.get('qualifications', []))}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
