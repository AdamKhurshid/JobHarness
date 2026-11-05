"""
Uses gemma4:12b (served via Ollama) to format the scraped jobs into a consisent JSON Structure
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request


# --------------------------------------------------------------------------- #
# Ollama client
# --------------------------------------------------------------------------- #
def _post(host: str, path: str, payload: dict, timeout: int) -> dict:
    url = f"{host.rstrip('/')}{path}"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def unload_model(host: str, model: str, timeout: int = 60) -> None:
    """Tell Ollama to release the model from memory (keep_alive: 0)."""
    try:
        _post(host, "/api/generate",
              {"model": model, "prompt": "", "keep_alive": 0}, timeout)
    except Exception as e:
        # Unloading is best-effort; don't fail the run if it errors.
        print(f"  (warn: unload request failed: {e})", file=sys.stderr)


class OllamaResponseError(RuntimeError):
    """Raised when Ollama returns a response that isn't usable JSON."""
    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw  # the raw model output, for debugging


def chat_with_ollama(host: str, model: str, system: str, user: str,
                     timeout: int = 180, retries: int = 4,
                     keep_alive: str = "5m", num_ctx: int = 8192,
                     backoff_base: float = 5.0,
                     force_json: bool = True,
                     debug_log=None) -> str:
    """Call Ollama's /api/chat endpoint with retry + backoff.

    force_json=True sends Ollama `format: "json"`, which uses llama.cpp's
    grammar-constrained sampling to FORCE valid JSON output. This is the
    cure for Gemma 3 returning prose instead of JSON.

    On network errors we pause (exponential backoff) and, from the 2nd retry
    on, ask Ollama to unload the model first — this clears VRAM if Ollama was
    OOM-adjacent, at the cost of a reload on the next call.
    """
    url = f"{host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "keep_alive": keep_alive,
        # The key fix: force grammar-constrained JSON output.
        "format": "json" if force_json else None,
        "options": {
            "temperature": 0.0,
            "num_predict": 1024,
            "num_ctx": num_ctx,
        },
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }
    # Drop the format key if None (Ollama rejects null format in some versions).
    if payload["format"] is None:
        del payload["format"]

    body = json.dumps(payload).encode("utf-8")

    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = (data.get("message") or {}).get("content", "").strip()
                if debug_log is not None:
                    debug_log.write(
                        f"=== attempt {attempt + 1} ===\n{content}\n\n"
                    )
                if not content:
                    raise OllamaResponseError(
                        "Empty response from model", raw=content
                    )
                return content
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError,
                json.JSONDecodeError, TimeoutError, ConnectionError) as e:
            last_err = e
            if attempt >= retries:
                break
            # Backoff: 5s, 10s, 20s, 40s ...
            wait = backoff_base * (2 ** attempt)
            print(f"  (retry {attempt + 1}/{retries} after {wait:.0f}s: "
                  f"{type(e).__name__}: {e})", file=sys.stderr)
            time.sleep(wait)
            # From the 2nd retry onward, force-unload the model to clear VRAM.
            if attempt >= 1:
                unload_model(host, model)
    raise RuntimeError(
        f"Ollama call failed after {retries + 1} tries: {last_err}"
    )


# --------------------------------------------------------------------------- #
# Prompt + JSON extraction
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are an information-extraction engine. You read a raw job listing and "
    "return ONE JSON object with EXACTLY these keys:\n"
    "  - company_name:   string\n"
    "  - title:          string\n"
    "  - url:            string\n"
    "  - pay_per_week:   string or null. Normalize the salary to a weekly amount "
    "when possible, e.g. \"$2,000/week\" or \"CNY 15,000/week\". Use null if not stated.\n"
    "  - qualifications: array of short strings (the must-have skills/requirements). "
    "Empty array if none are stated.\n"
    "  - location:       string (city, region, country). Empty string if not stated.\n"
    "  - is_in_person:   true if the role is on-site / in-person, false if remote / hybrid-as-remote, "
    "null if not stated.\n"
    "Rules:\n"
    "  - Output ONLY the JSON object. No prose, no markdown fences, no comments.\n"
    "  - If a field is missing from the source, use null (or empty array for qualifications).\n"
    "  - Preserve the original language of the listing (don't translate).\n"
)


def build_user_prompt(job: dict, max_desc_len: int = 8000) -> str:
    """Build the user prompt, truncating very long descriptions to fit context.

    Some job descriptions (especially scraped LinkedIn pages) can be 10KB+,
    which bloats the prompt and risks truncating the model's output. We cap
    the description field and keep the structured metadata fields intact.
    """
    # Work on a copy so we don't mutate the input.
    job_clean = dict(job)
    desc = job_clean.get("description", "")
    if isinstance(desc, str) and len(desc) > max_desc_len:
        job_clean["description"] = desc[:max_desc_len] + "\n...[TRUNCATED]..."
    return (
        "Extract the structured fields from this raw job listing.\n\n"
        f"RAW LISTING (JSON):\n{json.dumps(job_clean, ensure_ascii=False, indent=2)}\n\n"
        "Return ONLY a JSON object with keys: "
        "company_name, title, url, pay_per_week, qualifications, location, is_in_person. "
        "Start your response with { and end with }."
    )


def extract_json_object(text: str) -> dict:
    """Robustly pull the first {...} block out of an LLM response.

    Handles, in order:
      1. ```json ... ``` fenced code blocks
      2. <json>...</json> style XML/HTML tags (some Gemma versions use these)
      3. Bare balanced {...} object anywhere in the text
    """
    # Case 1: ```json ... ``` or ``` ... ``` fenced blocks.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return json.loads(fence.group(1))

    # Case 2: <json>...</json> or similar XML-ish wrappers.
    xml_wrap = re.search(r"<(?:json|output|result)[^>]*>(\{.*?\})</(?:json|output|result)>",
                         text, re.DOTALL | re.IGNORECASE)
    if xml_wrap:
        return json.loads(xml_wrap.group(1))

    # Case 3: bare balanced {...} object.
    start = text.find("{")
    if start == -1:
        raise OllamaResponseError(
            f"No JSON object found in response (no '{{' character). "
            f"Response starts with: {text[:300]!r}",
            raw=text,
        )
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])
    raise OllamaResponseError(
        f"Unbalanced braces in response. Response starts with: {text[:300]!r}",
        raw=text,
    )


def normalize_record(rec: dict) -> dict:
    """Coerce LLM output to the exact schema with sensible defaults."""
    def as_str(v):
        return v.strip() if isinstance(v, str) else ("" if v is None else str(v))

    def as_bool(v):
        if isinstance(v, bool):
            return v
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("true", "yes", "in-person", "in person", "on-site", "onsite", "1"):
                return True
            if s in ("false", "no", "remote", "hybrid-remote", "0"):
                return False
        return None

    quals = rec.get("qualifications", [])
    if isinstance(quals, str):
        quals = [q.strip() for q in re.split(r"[\n;•\-]+", quals) if q.strip()]
    elif not isinstance(quals, list):
        quals = []
    quals = [str(q).strip() for q in quals if str(q).strip()]

    return {
        "company_name":   as_str(rec.get("company_name")),
        "title":          as_str(rec.get("title")),
        "url":            as_str(rec.get("url")),
        "pay_per_week":   rec.get("pay_per_week") if isinstance(rec.get("pay_per_week"), str) else None,
        "qualifications": quals,
        "location":       as_str(rec.get("location")),
        "is_in_person":   as_bool(rec.get("is_in_person")),
    }


# --------------------------------------------------------------------------- #
# Atomic incremental save
# --------------------------------------------------------------------------- #
def save_atomic(records: list, path: str) -> None:
    """Write the full records list atomically: temp file in same dir, then rename."""
    dirname = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=dirname)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)  # atomic on POSIX & Windows
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_existing(path: str) -> list:
    """Load an existing output file for resume. Returns [] if missing/invalid."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def index_done(records: list) -> dict:
    """Map url -> record for already-completed (non-error) jobs."""
    return {
        r.get("url"): r for r in records
        if isinstance(r, dict) and r.get("url") and "_error" not in r
    }


# --------------------------------------------------------------------------- #
# Per-job processing
# --------------------------------------------------------------------------- #
def process_one(job: dict, host: str, model: str, num_ctx: int,
                keep_alive: str, timeout: int, max_desc_len: int = 8000,
                debug_log=None) -> dict:
    """Run one job through Gemma and return the normalized record.

    Strategy:
      1. Call Ollama with format:"json" (grammar-constrained) — this is the
         primary fix for Gemma outputting prose instead of JSON.
      2. If the response still isn't usable JSON, retry ONCE without
         format:"json" (some Ollama/model combos reject the grammar constraint
         or produce empty output under it).
    Both attempts are logged to debug_log if provided.
    """
    user_prompt = build_user_prompt(job, max_desc_len=max_desc_len)

    # Attempt 1: grammar-constrained JSON.
    try:
        raw = chat_with_ollama(host, model, SYSTEM_PROMPT, user_prompt,
                               timeout=timeout, keep_alive=keep_alive,
                               num_ctx=num_ctx, force_json=True,
                               debug_log=debug_log)
        return normalize_record(extract_json_object(raw))
    except OllamaResponseError as e1:
        # Attempt 2: free-form, then extract.
        if debug_log is not None:
            debug_log.write(
                f"--- fallback (no grammar constraint) after: {e1}\n"
            )
        raw = chat_with_ollama(host, model, SYSTEM_PROMPT, user_prompt,
                               timeout=timeout, keep_alive=keep_alive,
                               num_ctx=num_ctx, force_json=False,
                               debug_log=debug_log)
        return normalize_record(extract_json_object(raw))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Format job listings via Ollama + Gemma. "
                    "Stable for hundreds of jobs: saves after each job, resumable."
    )
    ap.add_argument("input", default="scraped_jobs.json")
    ap.add_argument("-o", "--output", default="formatted_jobs.json")
    ap.add_argument("--model", default="gemma4:12b",
                    help="Ollama model tag (default: gemma4:12b). "
                         "For low-RAM machines try gemma4:4b (~3 GB).")
    ap.add_argument("--host", default="http://localhost:11434",
                    help="Ollama base URL (default: http://localhost:11434).")
    ap.add_argument("--timeout", type=int, default=180,
                    help="Per-request timeout in seconds (default: 180).")
    ap.add_argument("--num-ctx", type=int, default=8192,
                    help="Model context window (default: 8192). "
                         "Lower this (e.g. 4096) to reduce KV-cache memory.")
    ap.add_argument("--keep-alive", default="5m",
                    help="How long Ollama keeps the model loaded after a call "
                         "(default: 5m). Use '0' to unload immediately after each call.")
    ap.add_argument("--unload-every", type=int, default=0,
                    help="Force-unload the model every N jobs to free VRAM "
                         "(default: 0 = off). e.g. --unload-every 10.")
    ap.add_argument("--unload-between", action="store_true",
                    help="Unload the model after EVERY call (slowest, most memory-safe).")
    ap.add_argument("--resume", action="store_true",
                    help="Skip jobs already present in the output file (matched by url).")
    ap.add_argument("--retry", type=int, default=4,
                    help="Retries per job on Ollama failure (default: 4).")
    ap.add_argument("--skip-errors", action="store_true",
                    help="With --resume, also skip jobs that previously errored "
                         "(by default errored jobs are retried).")
    ap.add_argument("--max-desc-len", type=int, default=8000,
                    help="Truncate the description field to this many chars "
                         "before sending to the model (default: 8000). "
                         "Prevents long scraped pages from blowing the context window.")
    ap.add_argument("--debug", metavar="FILE", default="",
                    help="Write every raw model response to this file for "
                         "debugging. e.g. --debug debug.log")
    args = ap.parse_args()

    # Load input -------------------------------------------------------------
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        sys.exit(f"Input file not found: {args.input}")
    except json.JSONDecodeError as e:
        sys.exit(f"Input is not valid JSON: {e}")

    jobs = data if isinstance(data, list) else [data]
    if not jobs:
        sys.exit("Input contains no jobs.")

    # Resume -----------------------------------------------------------------
    results = load_existing(args.output) if args.resume else []
    done = index_done(results) if args.resume else {}

    # If resuming, also honour --skip-errors: index errored urls too.
    errored = set()
    if args.resume and args.skip_errors:
        errored = {
            r.get("url") for r in results
            if isinstance(r, dict) and r.get("url") and "_error" in r
        }

    # We will rebuild `results` as a list aligned to the input order, but
    # preserve any extra records from a previous run that aren't in this input.
    print(f"Loaded {len(jobs)} job(s). Model: {args.model}  Host: {args.host}",
          file=sys.stderr)
    if args.resume:
        print(f"Resume: {len(done)} already done, {len(errored)} previously errored.",
              file=sys.stderr)

    # Build an output list aligned to input order.
    # Start from scratch for the input jobs; carry over completed records.
    out_records = [None] * len(jobs)
    carried_over_extra = []

    # Place already-completed records in their slots.
    for i, job in enumerate(jobs):
        u = job.get("url")
        if args.resume and u in done:
            out_records[i] = done[u]
        elif args.resume and u in errored:
            # Will be retried unless --skip-errors, in which case keep the error.
            out_records[i] = next(
                (r for r in results if isinstance(r, dict) and r.get("url") == u),
                None,
            )

    # Carry over any records in the existing output whose url is NOT in this
    # input (so we don't silently drop them).
    if args.resume:
        input_urls = {j.get("url") for j in jobs if j.get("url")}
        for r in results:
            if isinstance(r, dict) and r.get("url") and r["url"] not in input_urls:
                carried_over_extra.append(r)

    # Process ----------------------------------------------------------------
    debug_log = None
    if args.debug:
        debug_log = open(args.debug, "a", encoding="utf-8")
        debug_log.write(f"\n\n========== RUN {time.strftime('%Y-%m-%d %H:%M:%S')} ==========\n")
        print(f"Debug log: {args.debug}", file=sys.stderr)

    processed = 0
    skipped = 0
    try:
        for i, job in enumerate(jobs):
            # Skip if already done (and not an error we want to retry).
            if out_records[i] is not None and "_error" not in (out_records[i] or {}):
                skipped += 1
                print(f"[{i + 1}/{len(jobs)}] SKIP (already done): "
                      f"{job.get('company_name') or '?'} - {job.get('title') or '?'}",
                      file=sys.stderr)
                continue

            name = job.get("company_name") or "?"
            title = job.get("title") or "?"
            print(f"[{i + 1}/{len(jobs)}] Processing: {name} - {title}",
                  file=sys.stderr)

            if debug_log is not None:
                debug_log.write(
                    f"\n>>> JOB {i + 1}/{len(jobs)}: {name} - {title} "
                    f"({job.get('url')})\n"
                )

            try:
                rec = process_one(job, args.host, args.model, args.num_ctx,
                                  args.keep_alive, args.timeout,
                                  max_desc_len=args.max_desc_len,
                                  debug_log=debug_log)
                status = "ok"
            except Exception as e:
                rec = normalize_record(job)
                # Include a snippet of the raw response in the error if we have it.
                raw_snippet = ""
                if isinstance(e, OllamaResponseError) and e.raw:
                    raw_snippet = f" | raw: {e.raw[:200]!r}"
                rec["_error"] = f"{type(e).__name__}: {e}{raw_snippet}"
                status = f"FAILED ({type(e).__name__})"
                if debug_log is not None:
                    debug_log.write(f"!!! FAILED: {e}\nraw={e.raw if isinstance(e, OllamaResponseError) else ''}\n")
            out_records[i] = rec
            processed += 1
            print(f"           -> {status}", file=sys.stderr)

            # Periodic forced unload to bound VRAM usage.
            if args.unload_between:
                unload_model(args.host, args.model)
            elif args.unload_every > 0 and processed % args.unload_every == 0:
                print(f"           (unloading model to free memory)", file=sys.stderr)
                unload_model(args.host, args.model)

            # ---- Save after EVERY job (atomic) ----
            all_records = [r for r in out_records if r is not None] + carried_over_extra
            save_atomic(all_records, args.output)
    finally:
        if debug_log is not None:
            debug_log.close()

    # Final save (also catches the all-skipped case).
    all_records = [r for r in out_records if r is not None] + carried_over_extra
    save_atomic(all_records, args.output)

    ok = sum(1 for r in out_records if r and "_error" not in r)
    failed = sum(1 for r in out_records if r and "_error" in r)
    print(
        f"\nDone. Processed {processed} new, skipped {skipped}, "
        f"ok {ok}, failed {failed} (of {len(jobs)} input jobs).\n"
        f"Output: {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()