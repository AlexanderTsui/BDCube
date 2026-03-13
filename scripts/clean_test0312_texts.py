from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3.2"
DEFAULT_API_KEY_ENV = "SILICONFLOW_API_KEY"
DEFAULT_TIMEOUT = 240

SYSTEM_PROMPT = (
    "Rewrite the caption into one short English sentence for geometry-only 3D training. "
    "Keep only object category, silhouette, parts, structure, counts, and pose. "
    "Remove every color, material, texture, logo, printed text, weathering, and reflection detail. "
    "If a part is important but described with color or material, keep the part and delete only the appearance word. "
    "Example: 'a red car with blue mirrors' becomes 'a car with mirrors'. "
    "Return only the rewritten sentence."
)

FORBIDDEN_APPEARANCE_TERMS = (
    "black",
    "white",
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "gray",
    "grey",
    "silver",
    "gold",
    "beige",
    "peach",
    "bronze",
    "transparent",
    "translucent",
    "reflective",
    "glossy",
    "shiny",
    "metallic",
    "wooden",
    "stone",
    "rusty",
    "weathered",
    "worn",
    "patina",
    "camouflage",
    "logo",
    "logos",
    "emblem",
    "emblems",
    "lettering",
    "markings",
    "printed",
    "symbol",
    "symbols",
    "insignia",
    "embroidery",
    "textured",
    "texture",
)

FORBIDDEN_FILLER_TERMS = (
    "text",
    "texts",
    "label",
    "labels",
    "information",
    "info",
    "fact",
    "facts",
    "instruction",
    "instructions",
    "picture",
    "pictures",
    "description",
    "descriptions",
    "barcode",
    "details",
    "detail",
    "accents",
    "accent",
    "finish",
    "decorative",
    "decoration",
    "decorations",
    "number",
    "numbers",
    "letter",
    "letters",
    "stripe",
    "stripes",
)


@dataclass(frozen=True)
class RecordTask:
    index: int
    path: Path
    payload: dict[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean dataset/test0312 captions into geometry-only descriptions via SiliconFlow."
    )
    parser.add_argument(
        "--records-dir",
        type=str,
        default="dataset/test0312/records",
        help="Directory containing per-sample JSON records to clean.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="dataset/test0312_text_clean",
        help="Directory to store cleaned records, manifests, and logs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="SiliconFlow model name.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="SiliconFlow OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default=DEFAULT_API_KEY_ENV,
        help="Environment variable containing the SiliconFlow API key.",
    )
    parser.add_argument(
        "--api-key-file",
        type=str,
        default=None,
        help="Optional file containing the SiliconFlow API key.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum number of records to process. Use 0 or a negative value for all records.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N sorted records before processing.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of concurrent API workers.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Maximum retries per record for transient API failures.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the cleaning model.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.1,
        help="Nucleus sampling parameter for the cleaning model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=96,
        help="Maximum output tokens for the cleaning model.",
    )
    parser.add_argument(
        "--verify-model",
        action="store_true",
        help="Call GET /models before processing and warn if the requested model is not visible.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Re-run cleaning even if a cleaned record already exists.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=20,
        help="Number of cleaned examples to include in preview.json.",
    )
    return parser.parse_args()


def _load_api_key(args: argparse.Namespace) -> str:
    key = os.environ.get(args.api_key_env)
    if key:
        return key.strip()

    if args.api_key_file is not None:
        file_path = Path(args.api_key_file).expanduser().resolve()
        key = file_path.read_text().strip()
        if key:
            return key

    raise RuntimeError(
        "SiliconFlow API key not found. "
        f"Set {args.api_key_env} or provide --api-key-file."
    )


def _http_json(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    payload: Optional[dict[str, Any]],
    timeout: int,
) -> tuple[dict[str, Any], dict[str, str], int]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib_request.Request(url=url, data=data, headers=headers, method=method)
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            response_headers = {key.lower(): value for key, value in response.headers.items()}
            status = int(getattr(response, "status", response.getcode()))
    except urllib_error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        message = error_body
        try:
            payload_json = json.loads(error_body)
            message = json.dumps(payload_json, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Request to {url} failed: {exc}") from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from {url}: {body[:500]}") from exc
    return parsed, response_headers, status


def _list_models(
    *,
    api_key: str,
    base_url: str,
    timeout: int,
) -> list[str]:
    query = urllib_parse.urlencode({"type": "text", "sub_type": "chat"})
    url = f"{base_url.rstrip('/')}/models?{query}"
    payload, _, _ = _http_json(
        method="GET",
        url=url,
        headers={"Authorization": f"Bearer {api_key}"},
        payload=None,
        timeout=timeout,
    )
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected /models payload: {payload}")
    model_ids: list[str] = []
    for record in data:
        if isinstance(record, dict) and isinstance(record.get("id"), str):
            model_ids.append(record["id"])
    return model_ids


def _load_tasks(records_dir: Path, max_samples: int, offset: int) -> list[RecordTask]:
    record_paths = sorted(records_dir.glob("*.json"))
    if offset > 0:
        record_paths = record_paths[offset:]
    if max_samples > 0:
        record_paths = record_paths[:max_samples]

    tasks: list[RecordTask] = []
    for index, path in enumerate(record_paths):
        payload = json.loads(path.read_text())
        tasks.append(RecordTask(index=index, path=path, payload=payload))
    return tasks


def _clean_record_output_path(output_root: Path, task: RecordTask) -> Path:
    return output_root / "records" / task.path.name


def _build_user_prompt(
    record: dict[str, Any],
    *,
    invalid_terms: Optional[list[str]] = None,
    retry_feedback: Optional[str] = None,
) -> str:
    primary_caption = str(record["text"]).strip()
    if not invalid_terms and not retry_feedback:
        return primary_caption
    suffix_parts: list[str] = []
    if invalid_terms:
        invalid_text = ", ".join(invalid_terms)
        suffix_parts.append(
            "Your previous rewrite still contained forbidden appearance terms: "
            f"{invalid_text}. Remove them completely while keeping the geometry nouns."
        )
    if retry_feedback:
        suffix_parts.append(retry_feedback.strip())
    suffix_parts.append(
        "Rewrite again as one complete sentence. Keep the object category and geometric parts. "
        "Do not end with a dangling article like 'a' or 'an'. Return only the rewritten sentence."
    )
    return f"{primary_caption}\n\n" + " ".join(suffix_parts)


def _find_forbidden_terms(text: str) -> list[str]:
    normalized = text.lower()
    found: list[str] = []
    for term in (*FORBIDDEN_APPEARANCE_TERMS, *FORBIDDEN_FILLER_TERMS):
        if re.search(rf"\b{re.escape(term.lower())}\b", normalized):
            found.append(term)
    return found


def _normalize_clean_text(text: str) -> str:
    cleaned = " ".join(str(text).split())
    cleanup_patterns = (
        (r",\s*\.", "."),
        (r"\.\s*,", "."),
        (r"\s+,", ","),
        (r"\s+\.", "."),
        (r"\s{2,}", " "),
    )
    for pattern, replacement in cleanup_patterns:
        cleaned = re.sub(pattern, replacement, cleaned)
    cleaned = cleaned.strip(" ,.")
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    if cleaned and not cleaned.endswith("."):
        cleaned += "."
    return cleaned


def _find_malformed_issues(text: str) -> list[str]:
    normalized = _normalize_clean_text(text)
    lowered = normalized.lower()
    issues: list[str] = []
    malformed_patterns = (
        ("dangling_and_a", r"\band a\.$"),
        ("dangling_and_an", r"\band an\.$"),
        ("dangling_with_a", r"\bwith a\.$"),
        ("dangling_with_an", r"\bwith an\.$"),
        ("dangling_article_comma", r"\b(a|an),"),
        ("dangling_with_article_comma", r"\bwith\s+(a|an),"),
        ("dangling_and_article_comma", r"\band\s+(a|an),"),
        ("starts_with_a_with", r"^a with\b"),
        ("starts_with_an_with", r"^an with\b"),
        ("starts_with_the_with", r"^the with\b"),
        ("dangling_article", r"\b(a|an)\.$"),
        ("dangling_descriptor", r"\b(a|an)\s+(underside|lateral)\b"),
        ("restart_clause", r";\s*features?\b"),
        ("dangling_each_clause", r"\bon each [^,]+,\s+and\b"),
    )
    for issue, pattern in malformed_patterns:
        if re.search(pattern, lowered):
            issues.append(issue)
    if normalized.count(".") > 1:
        issues.append("multiple_sentences")
    return issues


def _scrub_forbidden_appearance_terms(text: str) -> str:
    cleaned = text
    for term in (*FORBIDDEN_APPEARANCE_TERMS, *FORBIDDEN_FILLER_TERMS):
        cleaned = re.sub(rf"\b{re.escape(term)}\b", "", cleaned, flags=re.IGNORECASE)

    cleanup_patterns = (
        (r"\bwith\s+and\b", "with"),
        (r"\band\s+and\b", "and"),
        (r"\bwith\s+with\b", "with"),
        (r"\bbody\s+with\b", "body"),
        (r"\bwith\s+(?=[,\.])", ""),
        (r"\band\s+(?=[,\.])", ""),
        (r"\ba\s+body\b", "body"),
        (r"\ba\s+prominent\b", "prominent"),
        (r",\s*,+", ", "),
        (r"\s+,", ","),
        (r"\s+\.", "."),
        (r"\s{2,}", " "),
    )
    for pattern, replacement in cleanup_patterns:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    return _normalize_clean_text(cleaned)


def _parse_cleaning_response(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()

    if not text:
        raise ValueError("Model returned empty cleaned text")

    if text.startswith("{"):
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object, got: {payload!r}")
        clean_text = payload.get("clean_text")
        if not isinstance(clean_text, str) or not clean_text.strip():
            raise ValueError(f"Response missing non-empty clean_text: {payload!r}")
        return {
            "clean_text": clean_text.strip(),
            "removed_appearance": [
                str(item).strip()
                for item in payload.get("removed_appearance", [])
                if str(item).strip()
            ],
            "notes": str(payload.get("notes", "")).strip(),
        }

    return {
        "clean_text": " ".join(text.split()),
        "removed_appearance": [],
        "notes": "",
    }


def _clean_single_record(
    *,
    task: RecordTask,
    output_root: Path,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
    max_retries: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    overwrite_output: bool,
    write_lock: threading.Lock,
) -> tuple[str, dict[str, Any]]:
    output_path = _clean_record_output_path(output_root, task)
    retry_feedback: Optional[str] = None
    if output_path.exists() and not overwrite_output:
        cached = json.loads(output_path.read_text())
        cached_clean_text = _normalize_clean_text(str(cached.get("clean_text", "")))
        cached["clean_text"] = cached_clean_text
        cached_invalid_terms = _find_forbidden_terms(cached_clean_text)
        cached_malformed_issues = _find_malformed_issues(cached_clean_text)
        if not cached_invalid_terms and not cached_malformed_issues:
            if cached_clean_text != str(json.loads(output_path.read_text()).get("clean_text", "")):
                with write_lock:
                    output_path.write_text(json.dumps(cached, indent=2, ensure_ascii=False))
            return "cached", cached
        feedback_parts: list[str] = []
        if cached_invalid_terms:
            feedback_parts.append(
                "The previous rewrite still included forbidden appearance terms: "
                + ", ".join(cached_invalid_terms)
                + "."
            )
        if cached_malformed_issues:
            feedback_parts.append(
                "The previous rewrite was malformed or incomplete: "
                + ", ".join(cached_malformed_issues)
                + "."
            )
        retry_feedback = " ".join(feedback_parts)

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    request_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(task.payload)},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "stream": False,
    }

    attempt = 0
    last_error: Optional[Exception] = None
    invalid_terms: Optional[list[str]] = None
    while attempt < max_retries:
        attempt += 1
        try:
            request_payload["messages"][1]["content"] = _build_user_prompt(
                task.payload,
                invalid_terms=invalid_terms,
                retry_feedback=retry_feedback,
            )
            response_payload, response_headers, _ = _http_json(
                method="POST",
                url=url,
                headers=headers,
                payload=request_payload,
                timeout=timeout,
            )
            choices = response_payload.get("choices")
            if not isinstance(choices, list) or not choices:
                raise RuntimeError(f"Invalid completion payload: {response_payload}")
            message = choices[0].get("message")
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise RuntimeError(f"Completion missing message content: {response_payload}")

            cleaned = _parse_cleaning_response(message["content"])
            cleaned["clean_text"] = _normalize_clean_text(cleaned["clean_text"])
            invalid_terms = _find_forbidden_terms(cleaned["clean_text"])
            if invalid_terms:
                scrubbed_text = _scrub_forbidden_appearance_terms(cleaned["clean_text"])
                scrubbed_invalid_terms = _find_forbidden_terms(scrubbed_text)
                scrubbed_malformed_issues = _find_malformed_issues(scrubbed_text)
                if not scrubbed_invalid_terms and not scrubbed_malformed_issues:
                    cleaned["clean_text"] = scrubbed_text
                    cleaned["notes"] = (
                        (cleaned["notes"] + "; " if cleaned["notes"] else "")
                        + "rule_scrubbed_forbidden_terms"
                    )
                else:
                    retry_feedback = (
                        "The previous rewrite still contained forbidden terms or became malformed "
                        "after removing them. Keep the object category, remove only appearance words, "
                        "and return one complete sentence."
                    )
                    raise ValueError(
                        "Cleaned text still contains forbidden appearance terms: "
                        + ", ".join(invalid_terms)
                    )
            malformed_issues = _find_malformed_issues(cleaned["clean_text"])
            if malformed_issues:
                retry_feedback = (
                    "The previous rewrite was malformed or incomplete: "
                    + ", ".join(malformed_issues)
                    + ". Keep the object noun and return one complete sentence."
                )
                raise ValueError(
                    "Cleaned text is malformed: " + ", ".join(malformed_issues)
                )
            result = {
                "item_id": task.payload.get("item_id"),
                "source_key": task.payload.get("source_key"),
                "mesh_path": task.payload.get("mesh_path"),
                "record_path": str(task.path.resolve()),
                "text": task.payload.get("text"),
                "clean_text": cleaned["clean_text"],
                "removed_appearance": cleaned["removed_appearance"],
                "notes": cleaned["notes"],
                "all_captions": task.payload.get("all_captions"),
                "bbox_xyz": task.payload.get("bbox_xyz"),
                "shape_token_count": task.payload.get("shape_token_count"),
                "text_seq_len": task.payload.get("text_seq_len"),
                "model": model,
                "usage": response_payload.get("usage"),
                "trace_id": response_headers.get("x-siliconcloud-trace-id"),
                "attempt": attempt,
            }
            with write_lock:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            return "cleaned", result
        except Exception as error:
            last_error = error
            backoff = min(2**attempt, 30) + random.random()
            logging.warning(
                "Clean failed for %s on attempt %d/%d: %s",
                task.path.name,
                attempt,
                max_retries,
                error,
            )
            if attempt < max_retries:
                time.sleep(backoff)

    assert last_error is not None
    raise RuntimeError(f"Failed to clean {task.path.name}: {last_error}") from last_error


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    records_dir = Path(args.records_dir).expanduser().resolve()
    if not records_dir.exists():
        raise FileNotFoundError(f"Records directory not found: {records_dir}")

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    api_key = _load_api_key(args)

    if args.verify_model:
        visible_models = _list_models(
            api_key=api_key,
            base_url=args.base_url,
            timeout=args.timeout,
        )
        if args.model not in visible_models:
            logging.warning(
                "Requested model %s is not visible in /models. Visible text-chat models: %s",
                args.model,
                visible_models[:50],
            )
        else:
            logging.info("Verified model visibility: %s", args.model)

    tasks = _load_tasks(
        records_dir=records_dir,
        max_samples=args.max_samples,
        offset=args.offset,
    )
    if not tasks:
        raise RuntimeError(f"No record files found under {records_dir}")

    write_lock = threading.Lock()
    cleaned_results: list[Optional[dict[str, Any]]] = [None] * len(tasks)
    failures: list[dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.num_workers, 1)) as executor:
        future_to_task = {
            executor.submit(
                _clean_single_record,
                task=task,
                output_root=output_root,
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                timeout=args.timeout,
                max_retries=args.max_retries,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                overwrite_output=args.overwrite_output,
                write_lock=write_lock,
            ): task
            for task in tasks
        }

        completed = 0
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                status, result = future.result()
                cleaned_results[task.index] = result
                completed += 1
                logging.info(
                    "[%d/%d] %s %s -> %s",
                    completed,
                    len(tasks),
                    status,
                    task.path.name,
                    result["clean_text"],
                )
            except Exception as error:
                completed += 1
                failure = {
                    "record_path": str(task.path.resolve()),
                    "item_id": task.payload.get("item_id"),
                    "text": task.payload.get("text"),
                    "error": repr(error),
                }
                failures.append(failure)
                logging.error("[%d/%d] failed %s: %s", completed, len(tasks), task.path.name, error)

    successful_results = [result for result in cleaned_results if result is not None]
    successful_results = [dict(result) for result in successful_results]

    clean_manifest_rows: list[dict[str, Any]] = []
    for result in successful_results:
        clean_manifest_rows.append(
            {
                "item_id": result.get("item_id"),
                "mesh_path": result.get("mesh_path"),
                "text": result.get("clean_text"),
                "original_text": result.get("text"),
                "clean_record_path": str(
                    (output_root / "records" / Path(result["record_path"]).name).resolve()
                ),
                "source_record_path": result.get("record_path"),
            }
        )

    preview_rows = [
        {
            "item_id": result.get("item_id"),
            "original_text": result.get("text"),
            "clean_text": result.get("clean_text"),
            "removed_appearance": result.get("removed_appearance"),
            "notes": result.get("notes"),
        }
        for result in successful_results[: max(args.preview_count, 0)]
    ]

    summary = {
        "records_dir": str(records_dir),
        "output_root": str(output_root),
        "model": args.model,
        "base_url": args.base_url,
        "requested_count": len(tasks),
        "cleaned_count": len(successful_results),
        "failed_count": len(failures),
        "clean_manifest_path": str((output_root / "clean_manifest.jsonl").resolve()),
        "preview_path": str((output_root / "preview.json").resolve()),
    }

    _write_jsonl(output_root / "clean_manifest.jsonl", clean_manifest_rows)
    _write_json(output_root / "preview.json", preview_rows)
    _write_json(output_root / "summary.json", summary)
    _write_json(output_root / "failures.json", failures)

    logging.info("Cleaning finished: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
