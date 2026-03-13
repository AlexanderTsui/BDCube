#!/usr/bin/env python3
"""
Resumable TRELLIS-500K downloader.

This tool is designed for long-running remote jobs:

1. `launch` starts a detached worker in a new session so the download keeps
   running after your SSH client disconnects.
2. `run` runs the worker in the foreground.
3. `status` summarizes progress from the manifest and append-only progress log.

The default layout matches this repository:

    ./dataset/minor_dataset   -> 4000 captioned samples for overfit experiments
    ./dataset/entire          -> captioned TRELLIS-500K training set

Notes:
- By default, only assets with non-empty `captions` are included.
- `3D-FUTURE` still requires you to place `3D-FUTURE-model.zip` manually under
  `<output_root>/3d_future/raw/`.
- `Toys4k` is an evaluation set and is excluded by default. Add
  `--include-toys4k` if you want it.
- `HSSD` may require a Hugging Face token with dataset access.
- `ObjaverseXL` downloads use direct HTTP downloads. The GitHub split is fetched
  from raw GitHub URLs instead of cloning repositories.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import zipfile
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote, unquote, urlsplit

import pandas as pd
import requests
from urllib3.util.retry import Retry

try:
    from huggingface_hub import get_token
except Exception:  # pragma: no cover - dependency is declared in repo
    get_token = None


LOGGER = logging.getLogger("trellis500k_downloader")

DATASET_REPO_ID = "JeffreyXiang/TRELLIS-500K"
OBJAVERSE_REPO_ID = "allenai/objaverse"
HF_PRIMARY_ENDPOINT = "https://huggingface.co"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
DEFAULT_WEB_PROXY_PREFIX = os.environ.get(
    "TRELLIS500K_WEB_PROXY_PREFIX",
    "https://web-proxy.ziplab.co/netsecv1",
)
TRAINING_SOURCE_KEYS = (
    "objaverse_xl_sketchfab",
    "objaverse_xl_github",
    "abo",
    "3d_future",
    "hssd",
)
EVAL_SOURCE_KEYS = ("toys4k",)
SUCCESS_STATUSES = {"downloaded", "verified_existing"}
TERMINAL_FAILURE_STATUSES = {"missing", "modified"}


@dataclass(frozen=True)
class SourceSpec:
    key: str
    label: str
    csv_filename: str
    kind: str
    manual_archive_name: Optional[str] = None


SOURCE_SPECS: Dict[str, SourceSpec] = {
    "objaverse_xl_sketchfab": SourceSpec(
        key="objaverse_xl_sketchfab",
        label="ObjaverseXL (sketchfab)",
        csv_filename="ObjaverseXL_sketchfab.csv",
        kind="objaverse_sketchfab",
    ),
    "objaverse_xl_github": SourceSpec(
        key="objaverse_xl_github",
        label="ObjaverseXL (github)",
        csv_filename="ObjaverseXL_github.csv",
        kind="objaverse_github",
    ),
    "abo": SourceSpec(
        key="abo",
        label="ABO",
        csv_filename="ABO.csv",
        kind="abo",
    ),
    "3d_future": SourceSpec(
        key="3d_future",
        label="3D-FUTURE",
        csv_filename="3D-FUTURE.csv",
        kind="3d_future",
        manual_archive_name="3D-FUTURE-model.zip",
    ),
    "hssd": SourceSpec(
        key="hssd",
        label="HSSD",
        csv_filename="HSSD.csv",
        kind="hssd",
    ),
    "toys4k": SourceSpec(
        key="toys4k",
        label="Toys4k",
        csv_filename="Toys4k.csv",
        kind="toys4k",
        manual_archive_name="toys4k_blend_files.zip",
    ),
}


class SourceUnavailable(RuntimeError):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_output_root(job: str) -> Path:
    base = repo_root() / "dataset"
    if job == "minor":
        return base / "minor_dataset"
    if job == "entire":
        return base / "entire"
    raise ValueError(f"Unsupported job: {job}")


def manifest_path(output_root: Path) -> Path:
    return output_root / "manifest.csv.gz"


def progress_path(output_root: Path) -> Path:
    return output_root / "progress.jsonl"


def blocked_sources_path(output_root: Path) -> Path:
    return output_root / "blocked_sources.json"


def job_config_path(output_root: Path) -> Path:
    return output_root / "job_config.json"


def pid_path(output_root: Path) -> Path:
    return output_root / "download.pid"


def log_path(output_root: Path) -> Path:
    return output_root / "logs" / "trellis500k_downloader.log"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_item_id(source_key: str, sha256: str) -> str:
    return f"{source_key}:{sha256}"


def configure_logging(log_file: Path, detached: bool) -> None:
    ensure_dir(log_file.parent)
    handlers: List[logging.Handler] = [
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    if not detached:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def has_live_pid(pid_file: Path) -> Tuple[bool, Optional[int]]:
    if not pid_file.exists():
        return False, None
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except Exception:
        return False, None
    try:
        os.kill(pid, 0)
    except OSError:
        return False, pid
    return True, pid


def write_pid_file(pid_file: Path, pid: int) -> None:
    ensure_dir(pid_file.parent)
    pid_file.write_text(str(pid), encoding="utf-8")


def remove_pid_file(pid_file: Path) -> None:
    try:
        pid_file.unlink()
    except FileNotFoundError:
        pass


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def captions_mask(series: pd.Series) -> pd.Series:
    strings = series.fillna("").astype(str).str.strip()
    return (~strings.isin({"", "[]", "nan", "NaN", "None"}))


def parse_sources(job: str, sources: Optional[str], include_toys4k: bool) -> List[str]:
    if sources:
        keys = [part.strip() for part in sources.split(",") if part.strip()]
    else:
        keys = list(TRAINING_SOURCE_KEYS)
        if include_toys4k:
            keys.extend(EVAL_SOURCE_KEYS)
    unknown = [key for key in keys if key not in SOURCE_SPECS]
    if unknown:
        raise ValueError(f"Unknown source(s): {', '.join(unknown)}")
    if job == "minor" and "toys4k" in keys and not include_toys4k:
        raise ValueError("Use --include-toys4k if you want Toys4k in the minor job.")
    return keys


def detected_hf_token() -> Optional[str]:
    if get_token is None:
        return None
    token = get_token()
    if token:
        return token
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    return None


def hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", HF_PRIMARY_ENDPOINT).rstrip("/")


def unique_strings(values: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def apply_web_proxy(url: str, proxy_prefix: Optional[str]) -> str:
    if not proxy_prefix:
        return url
    prefix = proxy_prefix.rstrip("/")
    if url.startswith(prefix + "/"):
        return url
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        return url
    suffix = url[len(parsed.scheme) + 3 :]
    return f"{prefix}/{parsed.scheme}/{suffix}"


def direct_then_proxy_urls(url: str, proxy_prefix: Optional[str]) -> List[str]:
    candidates = [url]
    proxied = apply_web_proxy(url, proxy_prefix)
    if proxied != url:
        candidates.append(proxied)
    return unique_strings(candidates)


def preferred_hf_endpoints() -> List[str]:
    configured = hf_endpoint()
    candidates = [HF_MIRROR_ENDPOINT, configured, HF_PRIMARY_ENDPOINT]
    return unique_strings([endpoint.rstrip("/") for endpoint in candidates])


def hf_dataset_candidate_urls(repo_id: str, filename: str, proxy_prefix: Optional[str]) -> List[str]:
    path = f"/datasets/{repo_id}/resolve/main/{quote(filename, safe='/')}"
    candidates: List[str] = []
    for endpoint in preferred_hf_endpoints():
        candidates.extend(direct_then_proxy_urls(f"{endpoint}{path}", proxy_prefix))
    return unique_strings(candidates)


def github_blob_to_raw_urls(file_identifier: str, proxy_prefix: Optional[str]) -> List[str]:
    parsed = urlsplit(file_identifier)
    path_parts = parsed.path.lstrip("/").split("/")
    if parsed.netloc != "github.com" or len(path_parts) < 5 or path_parts[2] != "blob":
        raise ValueError(f"Unsupported GitHub blob URL: {file_identifier}")
    org, repo, _, commit = path_parts[:4]
    file_path = "/".join(quote(part, safe="") for part in path_parts[4:])
    raw_url = f"https://raw.githubusercontent.com/{org}/{repo}/{commit}/{file_path}"
    return direct_then_proxy_urls(raw_url, proxy_prefix)


def github_blob_relative_path(file_identifier: str) -> Path:
    parsed = urlsplit(file_identifier)
    path_parts = parsed.path.lstrip("/").split("/")
    if parsed.netloc != "github.com" or len(path_parts) < 5 or path_parts[2] != "blob":
        raise ValueError(f"Unsupported GitHub blob URL: {file_identifier}")
    org, repo, _, commit = path_parts[:4]
    decoded_parts = [unquote(part) for part in path_parts[4:]]
    return Path(org) / repo / commit / Path(*decoded_parts)


def source_root(output_root: Path, source_key: str) -> Path:
    return output_root / source_key


def manual_archive_path(output_root: Path, spec: SourceSpec) -> Optional[Path]:
    if not spec.manual_archive_name:
        return None
    return source_root(output_root, spec.key) / "raw" / spec.manual_archive_name


def source_readiness(
    output_root: Path,
    source_key: str,
    require_objaverse: bool = True,
) -> Tuple[bool, Optional[str]]:
    spec = SOURCE_SPECS[source_key]
    if spec.kind.startswith("objaverse"):
        return True, None
    if spec.kind == "hssd":
        if get_token is None:
            return False, "Missing `huggingface_hub`, required for HSSD downloads."
        if detected_hf_token() is None:
            return (
                False,
                "No Hugging Face token found. Set HF_TOKEN or run `huggingface-cli login` for HSSD.",
            )
    if spec.manual_archive_name:
        archive = manual_archive_path(output_root, spec)
        if archive is None or not archive.exists():
            return (
                False,
                f"Missing manual archive `{spec.manual_archive_name}` at `{archive}`.",
            )
    return True, None


def save_blocked_source(output_root: Path, source_key: str, reason: str) -> None:
    blocked = read_json(blocked_sources_path(output_root), default={})
    blocked[source_key] = reason
    write_json(blocked_sources_path(output_root), blocked)


def clear_blocked_source(output_root: Path, source_key: str) -> None:
    blocked = read_json(blocked_sources_path(output_root), default={})
    if source_key in blocked:
        blocked.pop(source_key)
        write_json(blocked_sources_path(output_root), blocked)


def load_source_metadata(
    output_root: Path,
    source_key: str,
    refresh: bool = False,
    proxy_prefix: Optional[str] = None,
) -> pd.DataFrame:
    spec = SOURCE_SPECS[source_key]
    local_copy = output_root / "metadata" / spec.csv_filename
    ensure_dir(local_copy.parent)
    if refresh or not local_copy.exists():
        download_url_resumable(
            hf_dataset_candidate_urls(DATASET_REPO_ID, spec.csv_filename, proxy_prefix),
            local_copy,
        )
    df = pd.read_csv(local_copy)
    df = df[["sha256", "file_identifier", "aesthetic_score", "captions"]].copy()
    df["source_key"] = source_key
    df["source_label"] = spec.label
    return df


def allocate_counts(source_frames: Dict[str, pd.DataFrame], sample_size: int) -> Dict[str, int]:
    counts = {key: len(df) for key, df in source_frames.items() if len(df) > 0}
    total = sum(counts.values())
    if total == 0:
        return {}
    if sample_size >= total:
        return counts

    keys = list(counts.keys())
    raw = {key: (counts[key] * sample_size / total) for key in keys}
    allocated = {key: int(raw[key]) for key in keys}
    remainder = sample_size - sum(allocated.values())
    for key in sorted(keys, key=lambda item: (raw[item] - allocated[item]), reverse=True):
        if remainder <= 0:
            break
        if allocated[key] < counts[key]:
            allocated[key] += 1
            remainder -= 1

    while remainder > 0:
        advanced = False
        for key in keys:
            if allocated[key] < counts[key]:
                allocated[key] += 1
                remainder -= 1
                advanced = True
                if remainder == 0:
                    break
        if not advanced:
            break
    return allocated


def build_manifest(
    output_root: Path,
    job: str,
    source_keys: Sequence[str],
    sample_size: int,
    seed: int,
    require_captions: bool,
    refresh_metadata: bool,
    available_only_for_minor: bool,
    proxy_prefix: Optional[str],
) -> pd.DataFrame:
    LOGGER.info("Building manifest for %s under %s", job, output_root)
    ready_reasons: Dict[str, str] = {}
    selected_sources: List[str] = []
    for key in source_keys:
        ready, reason = source_readiness(output_root, key)
        if job == "minor" and available_only_for_minor and not ready:
            LOGGER.warning("Skipping %s while building minor manifest: %s", key, reason)
            ready_reasons[key] = reason or "not ready"
            continue
        selected_sources.append(key)

    if not selected_sources:
        raise RuntimeError("No usable sources left for manifest generation.")

    source_frames: Dict[str, pd.DataFrame] = {}
    for key in selected_sources:
        df = load_source_metadata(
            output_root,
            key,
            refresh=refresh_metadata,
            proxy_prefix=proxy_prefix,
        )
        if require_captions:
            df = df[captions_mask(df["captions"])].copy()
        source_frames[key] = df
        LOGGER.info("Loaded %s rows from %s", len(df), key)

    if job == "minor":
        allocation = allocate_counts(source_frames, sample_size)
        sampled_frames: List[pd.DataFrame] = []
        for index, key in enumerate(selected_sources):
            frame = source_frames[key]
            target = allocation.get(key, 0)
            if target <= 0:
                continue
            sampled = frame.sample(n=target, random_state=seed + index) if target < len(frame) else frame.copy()
            sampled_frames.append(sampled)
        manifest = pd.concat(sampled_frames, ignore_index=True)
    else:
        manifest = pd.concat([source_frames[key] for key in selected_sources], ignore_index=True)

    manifest["item_id"] = [
        make_item_id(source_key, sha256)
        for source_key, sha256 in zip(manifest["source_key"], manifest["sha256"])
    ]
    manifest = manifest[
        ["item_id", "source_key", "source_label", "sha256", "file_identifier", "aesthetic_score", "captions"]
    ].sort_values(["source_key", "sha256"], kind="stable")
    manifest.to_csv(manifest_path(output_root), index=False, compression="gzip")

    job_config = {
        "job": job,
        "output_root": str(output_root),
        "source_keys": list(selected_sources),
        "requested_source_keys": list(source_keys),
        "sample_size": sample_size,
        "seed": seed,
        "require_captions": require_captions,
        "available_only_for_minor": available_only_for_minor,
        "manifest_rows": int(len(manifest)),
        "created_at": int(time.time()),
        "skipped_sources": ready_reasons,
    }
    write_json(job_config_path(output_root), job_config)
    LOGGER.info("Manifest written to %s with %d rows", manifest_path(output_root), len(manifest))
    return manifest


def load_manifest(
    output_root: Path,
    rebuild: bool,
    job: str,
    source_keys: Sequence[str],
    sample_size: int,
    seed: int,
    require_captions: bool,
    refresh_metadata: bool,
    available_only_for_minor: bool,
    proxy_prefix: Optional[str],
) -> pd.DataFrame:
    if rebuild or not manifest_path(output_root).exists():
        return build_manifest(
            output_root=output_root,
            job=job,
            source_keys=source_keys,
            sample_size=sample_size,
            seed=seed,
            require_captions=require_captions,
            refresh_metadata=refresh_metadata,
            available_only_for_minor=available_only_for_minor,
            proxy_prefix=proxy_prefix,
        )
    LOGGER.info("Loading existing manifest from %s", manifest_path(output_root))
    return pd.read_csv(manifest_path(output_root))


class ProgressTracker:
    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.path = progress_path(output_root)
        ensure_dir(self.path.parent)
        self._write_lock = None
        self.latest: Dict[str, Dict[str, object]] = {}
        self.failures: Counter = Counter()
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    item_id = event["item_id"]
                    self.latest[item_id] = event
                    if event["status"] == "failed":
                        self.failures[item_id] += 1

    def record(
        self,
        *,
        item_id: str,
        source_key: str,
        sha256: str,
        status: str,
        file_identifier: str,
        local_path: Optional[str] = None,
        member_path: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        event = {
            "ts": int(time.time()),
            "item_id": item_id,
            "source_key": source_key,
            "sha256": sha256,
            "status": status,
            "file_identifier": file_identifier,
        }
        if local_path:
            event["local_path"] = local_path
        if member_path:
            event["member_path"] = member_path
        if error:
            event["error"] = error

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

        self.latest[item_id] = event
        if status == "failed":
            self.failures[item_id] += 1

    def latest_status(self, item_id: str) -> Optional[str]:
        event = self.latest.get(item_id)
        return None if event is None else str(event["status"])

    def is_terminal(self, item_id: str) -> bool:
        status = self.latest_status(item_id)
        return status in SUCCESS_STATUSES or status in TERMINAL_FAILURE_STATUSES

    def failure_count(self, item_id: str) -> int:
        return int(self.failures.get(item_id, 0))


def format_bytes(num_bytes: float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0 or not seconds < float("inf"):
        return "unknown"
    total_seconds = int(round(seconds))
    if total_seconds < 60:
        return f"{total_seconds}s"
    minutes, secs = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, mins = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h{mins:02d}m"
    days, hrs = divmod(hours, 24)
    return f"{days}d{hrs:02d}h"


class TransferMeter:
    def __init__(self, label: str, total_items: int, log_interval_seconds: float) -> None:
        self.label = label
        self.total_items = total_items
        self.log_interval_seconds = max(5.0, float(log_interval_seconds))
        self._recent_window_seconds = 60.0
        self._lock = threading.Lock()
        self._started_at = time.monotonic()
        self._last_log_at = self._started_at
        self._total_bytes = 0
        self._interval_bytes = 0
        self._completed_items = 0
        self._byte_events = deque()
        self._item_events = deque()

    def add_bytes(self, byte_count: int) -> None:
        if byte_count <= 0:
            return
        self._log_if_needed(byte_count)

    def mark_item_complete(self) -> None:
        with self._lock:
            self._completed_items += 1
            now = time.monotonic()
            self._item_events.append(now)
            self._prune_events(now)

    def log_summary(self, force: bool = False) -> None:
        message = self._build_message(force=force)
        if message:
            LOGGER.info(message)

    def _log_if_needed(self, byte_count: int) -> None:
        message = None
        now = time.monotonic()
        with self._lock:
            self._total_bytes += byte_count
            self._interval_bytes += byte_count
            self._byte_events.append((now, byte_count))
            self._prune_events(now)
            elapsed = now - self._last_log_at
            if elapsed < self.log_interval_seconds:
                return
            message = self._render_message(now)
            self._interval_bytes = 0
            self._last_log_at = now
        LOGGER.info(message)

    def _build_message(self, force: bool) -> Optional[str]:
        now = time.monotonic()
        with self._lock:
            self._prune_events(now)
            elapsed = now - self._last_log_at
            if not force and elapsed < self.log_interval_seconds:
                return None
            if self._total_bytes <= 0:
                return (
                    f"{self.label} speed: items={self._completed_items}/{self.total_items} "
                    "current=0.0 B/s avg=0.0 B/s avg_1m=0.0 B/s eta=unknown transferred=0.0 B"
                )
            message = self._render_message(now)
            self._interval_bytes = 0
            self._last_log_at = now
            return message

    def _prune_events(self, now: float) -> None:
        cutoff = now - self._recent_window_seconds
        while self._byte_events and self._byte_events[0][0] < cutoff:
            self._byte_events.popleft()
        while self._item_events and self._item_events[0] < cutoff:
            self._item_events.popleft()

    def _recent_byte_rate(self, now: float) -> float:
        if not self._byte_events:
            return 0.0
        total = sum(byte_count for _, byte_count in self._byte_events)
        span = min(self._recent_window_seconds, max(now - self._started_at, 1e-6))
        return total / span

    def _eta_seconds(self, now: float) -> Optional[float]:
        remaining_items = max(self.total_items - self._completed_items, 0)
        if remaining_items == 0:
            return 0.0
        recent_count = len(self._item_events)
        if recent_count >= 1:
            recent_span = min(self._recent_window_seconds, max(now - self._started_at, 1e-6))
            recent_item_rate = recent_count / recent_span
            if recent_item_rate > 0:
                return remaining_items / recent_item_rate
        total_elapsed = max(now - self._started_at, 1e-6)
        avg_item_rate = self._completed_items / total_elapsed
        if avg_item_rate > 0:
            return remaining_items / avg_item_rate
        return None

    def _render_message(self, now: float) -> str:
        interval_elapsed = max(now - self._last_log_at, 1e-6)
        total_elapsed = max(now - self._started_at, 1e-6)
        current_rate = self._interval_bytes / interval_elapsed
        average_rate = self._total_bytes / total_elapsed
        recent_rate = self._recent_byte_rate(now)
        eta_seconds = self._eta_seconds(now)
        return (
            f"{self.label} speed: items={self._completed_items}/{self.total_items} "
            f"current={format_bytes(current_rate)}/s "
            f"avg={format_bytes(average_rate)}/s "
            f"avg_1m={format_bytes(recent_rate)}/s "
            f"eta={format_duration(eta_seconds)} "
            f"transferred={format_bytes(self._total_bytes)}"
        )


def summarize_progress(manifest: pd.DataFrame, tracker: ProgressTracker) -> Dict[str, Dict[str, int]]:
    total_by_source = Counter(manifest["source_key"])
    status_by_source: Dict[str, Counter] = defaultdict(Counter)
    for row in manifest.itertuples(index=False):
        status = tracker.latest_status(row.item_id) or "pending"
        status_by_source[row.source_key][status] += 1

    summary = {}
    for source_key, total in total_by_source.items():
        payload = {"total": int(total)}
        payload.update({status: int(count) for status, count in status_by_source[source_key].items()})
        summary[source_key] = payload
    return summary


def log_summary(manifest: pd.DataFrame, tracker: ProgressTracker) -> None:
    summary = summarize_progress(manifest, tracker)
    for source_key in manifest["source_key"].drop_duplicates().tolist():
        payload = summary.get(source_key, {"total": 0})
        LOGGER.info(
            "Status %-24s total=%-7d downloaded=%-7d verified=%-7d pending=%-7d failed=%-7d missing=%-7d modified=%-7d",
            source_key,
            payload.get("total", 0),
            payload.get("downloaded", 0),
            payload.get("verified_existing", 0),
            payload.get("pending", 0),
            payload.get("failed", 0),
            payload.get("missing", 0),
            payload.get("modified", 0),
        )


def parse_caption_list(raw_value: object) -> List[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return raw_value
    if isinstance(raw_value, float) and pd.isna(raw_value):
        return []
    text = str(raw_value).strip()
    if not text or text in {"nan", "NaN", "[]"}:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def export_downloaded_pairs(output_root: Path, manifest: pd.DataFrame, tracker: ProgressTracker) -> Path:
    export_path = output_root / "downloaded_pairs.jsonl"
    rows = []
    for row in manifest.itertuples(index=False):
        latest = tracker.latest.get(row.item_id)
        if not latest or latest["status"] not in SUCCESS_STATUSES:
            continue
        record = {
            "item_id": row.item_id,
            "source_key": row.source_key,
            "sha256": row.sha256,
            "file_identifier": row.file_identifier,
            "local_path": latest.get("local_path"),
            "captions": parse_caption_list(row.captions),
            "primary_caption": None,
            "aesthetic_score": row.aesthetic_score,
        }
        captions = record["captions"]
        if captions:
            record["primary_caption"] = captions[0]
        if latest.get("member_path"):
            record["member_path"] = latest["member_path"]
        rows.append(record)

    with export_path.open("w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    LOGGER.info("Exported %d downloaded text-shape pairs to %s", len(rows), export_path)
    return export_path


def relpath_to_output(output_root: Path, path: Path) -> str:
    return os.path.relpath(path, output_root)


def split_archive_member_path(resolved_path: str) -> Tuple[str, Optional[str]]:
    for suffix in (".tar.gz/", ".zip/", ".tar/"):
        marker = resolved_path.find(suffix)
        if marker >= 0:
            archive = resolved_path[: marker + len(suffix) - 1]
            member = resolved_path[marker + len(suffix) :]
            return archive, member
    return resolved_path, None


def make_requests_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def download_url_resumable(
    url: str | Sequence[str],
    destination: Path,
    timeout: int = 120,
    headers: Optional[Dict[str, str]] = None,
    meter: Optional[TransferMeter] = None,
) -> None:
    ensure_dir(destination.parent)
    partial = destination.with_suffix(destination.suffix + ".part")
    raw_candidates = [url] if isinstance(url, str) else [item for item in url if item]
    candidates = unique_strings(raw_candidates)
    errors: List[str] = []

    for index, candidate_url in enumerate(candidates, start=1):
        session = make_requests_session()
        try:
            if index > 1:
                LOGGER.warning(
                    "Retrying %s via fallback URL %d/%d: %s",
                    destination,
                    index,
                    len(candidates),
                    candidate_url,
                )
            while True:
                resume_from = partial.stat().st_size if partial.exists() else 0
                request_headers = dict(headers or {})
                if resume_from:
                    request_headers["Range"] = f"bytes={resume_from}-"
                with session.get(candidate_url, stream=True, headers=request_headers, timeout=timeout) as response:
                    if response.status_code == 416 and partial.exists():
                        partial.rename(destination)
                        return
                    response.raise_for_status()
                    status = response.status_code
                    if resume_from and status == 200:
                        partial.unlink(missing_ok=True)
                        continue
                    mode = "ab" if status == 206 and resume_from else "wb"
                    with partial.open(mode) as handle:
                        for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                                if meter is not None:
                                    meter.add_bytes(len(chunk))
                partial.rename(destination)
                return
        except Exception as exc:
            errors.append(f"{candidate_url}: {exc}")
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code in {404, 410}:
                raise
        finally:
            session.close()

    raise RuntimeError(
        f"All download candidates failed for {destination}: {' | '.join(errors[-4:])}"
    )


def direct_download_worker(
    *,
    url: str | Sequence[str],
    destination: Path,
    expected_sha256: str,
    headers: Optional[Dict[str, str]] = None,
    meter: Optional[TransferMeter] = None,
) -> Tuple[str, Path]:
    if destination.exists():
        existing_sha = sha256_file(destination)
        if existing_sha == expected_sha256:
            return "verified_existing", destination
        destination.unlink()

    download_url_resumable(url, destination, headers=headers, meter=meter)
    downloaded_sha = sha256_file(destination)
    if downloaded_sha != expected_sha256:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 mismatch for {destination}")
    return "downloaded", destination


def status_for_download_exception(exc: Exception) -> str:
    text = str(exc)
    if "404" in text or "410" in text:
        return "missing"
    return "failed"


def ensure_objaverse_sketchfab_object_paths(
    raw_dir: Path,
    proxy_prefix: Optional[str],
) -> Dict[str, str]:
    versioned_dir = raw_dir / "hf-objaverse-v1"
    ensure_dir(versioned_dir)
    destination = versioned_dir / "object-paths.json.gz"
    tmp_destination = destination.with_suffix(destination.suffix + ".tmp")
    if tmp_destination.exists() and not destination.exists():
        tmp_destination.unlink(missing_ok=True)

    if not destination.exists():
        LOGGER.info("Downloading Sketchfab object-paths index to %s", destination)
        download_url_resumable(
            hf_dataset_candidate_urls(OBJAVERSE_REPO_ID, "object-paths.json.gz", proxy_prefix),
            destination,
        )

    try:
        LOGGER.info("Loading Sketchfab object-paths index from %s", destination)
        with gzip.open(destination, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def download_objaverse_sketchfab(
    rows: pd.DataFrame,
    output_root: Path,
    tracker: ProgressTracker,
    workers: int,
    max_retries: int,
    proxy_prefix: Optional[str],
    speed_log_interval: float,
) -> None:
    source_key = "objaverse_xl_sketchfab"
    raw_dir = source_root(output_root, source_key) / "raw"
    ensure_dir(raw_dir)
    object_paths = ensure_objaverse_sketchfab_object_paths(raw_dir, proxy_prefix)

    pending = [row for row in rows.itertuples(index=False) if should_process_row(row, tracker, max_retries)]
    if not pending:
        LOGGER.info("No pending %s items.", source_key)
        return

    LOGGER.info("Downloading %d ObjaverseXL Sketchfab items with %d workers", len(pending), workers)
    completed = 0
    meter = TransferMeter("ObjaverseXL Sketchfab", len(pending), speed_log_interval)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_row = {}
        for row in pending:
            uid = row.file_identifier.rstrip("/").split("/")[-1]
            hf_object_path = object_paths.get(uid)
            if not hf_object_path:
                tracker.record(
                    item_id=row.item_id,
                    source_key=source_key,
                    sha256=row.sha256,
                    status="missing",
                    file_identifier=row.file_identifier,
                    error=f"uid {uid} not found in object-paths.json.gz",
                )
                completed += 1
                meter.mark_item_complete()
                continue

            destination = raw_dir / "hf-objaverse-v1" / hf_object_path
            url = hf_dataset_candidate_urls(OBJAVERSE_REPO_ID, hf_object_path, proxy_prefix)
            future = executor.submit(
                direct_download_worker,
                url=url,
                destination=destination,
                expected_sha256=row.sha256,
                meter=meter,
            )
            future_to_row[future] = row

        for future in as_completed(future_to_row):
            row = future_to_row[future]
            try:
                status, destination = future.result()
                tracker.record(
                    item_id=row.item_id,
                    source_key=source_key,
                    sha256=row.sha256,
                    status=status,
                    file_identifier=row.file_identifier,
                    local_path=relpath_to_output(output_root, destination),
                )
            except Exception as exc:
                tracker.record(
                    item_id=row.item_id,
                    source_key=source_key,
                    sha256=row.sha256,
                    status=status_for_download_exception(exc),
                    file_identifier=row.file_identifier,
                    error=str(exc),
                )
            completed += 1
            meter.mark_item_complete()
            if completed % 50 == 0 or completed == len(pending):
                LOGGER.info("ObjaverseXL Sketchfab progress: %d/%d", completed, len(pending))
                meter.log_summary()
    meter.log_summary(force=True)


def should_process_row(row, tracker: ProgressTracker, max_retries: int) -> bool:
    item_id = row["item_id"] if isinstance(row, pd.Series) else row.item_id
    if tracker.is_terminal(item_id):
        return False
    if tracker.failure_count(item_id) >= max_retries:
        return False
    return True


def download_abo(
    rows: pd.DataFrame,
    output_root: Path,
    tracker: ProgressTracker,
    workers: int,
    max_retries: int,
    proxy_prefix: Optional[str],
    speed_log_interval: float,
) -> None:
    source_key = "abo"
    base_dir = source_root(output_root, source_key) / "raw" / "3dmodels" / "original"
    ensure_dir(base_dir)
    pending = [row for row in rows.itertuples(index=False) if should_process_row(row, tracker, max_retries)]
    if not pending:
        LOGGER.info("No pending ABO items.")
        return

    LOGGER.info("Downloading %d ABO items with %d workers", len(pending), workers)
    completed = 0
    meter = TransferMeter("ABO", len(pending), speed_log_interval)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_row = {}
        for row in pending:
            destination = base_dir / row.file_identifier
            url = direct_then_proxy_urls(
                f"https://amazon-berkeley-objects.s3.amazonaws.com/3dmodels/original/{row.file_identifier}",
                proxy_prefix,
            )
            future = executor.submit(
                direct_download_worker,
                url=url,
                destination=destination,
                expected_sha256=row.sha256,
                meter=meter,
            )
            future_to_row[future] = row

        for future in as_completed(future_to_row):
            row = future_to_row[future]
            try:
                status, destination = future.result()
                tracker.record(
                    item_id=row.item_id,
                    source_key=source_key,
                    sha256=row.sha256,
                    status=status,
                    file_identifier=row.file_identifier,
                    local_path=relpath_to_output(output_root, destination),
                )
            except Exception as exc:
                tracker.record(
                    item_id=row.item_id,
                    source_key=source_key,
                    sha256=row.sha256,
                    status=status_for_download_exception(exc),
                    file_identifier=row.file_identifier,
                    error=str(exc),
                )
            completed += 1
            meter.mark_item_complete()
            if completed % 50 == 0 or completed == len(pending):
                LOGGER.info("ABO progress: %d/%d", completed, len(pending))
                meter.log_summary()
    meter.log_summary(force=True)


def download_hssd(
    rows: pd.DataFrame,
    output_root: Path,
    tracker: ProgressTracker,
    workers: int,
    max_retries: int,
    token: Optional[str],
    proxy_prefix: Optional[str],
    speed_log_interval: float,
) -> None:
    if get_token is None:
        raise SourceUnavailable("huggingface_hub is required to provide the HSSD token.")

    source_key = "hssd"
    base_dir = source_root(output_root, source_key) / "raw"
    ensure_dir(base_dir)
    pending = [row for row in rows.itertuples(index=False) if should_process_row(row, tracker, max_retries)]
    if not pending:
        LOGGER.info("No pending HSSD items.")
        return

    LOGGER.info("Downloading %d HSSD items with %d workers", len(pending), workers)
    completed = 0
    meter = TransferMeter("HSSD", len(pending), speed_log_interval)
    headers = {"Authorization": f"Bearer {token}"} if token else None
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_row = {}
        for row in pending:
            destination = base_dir / row.file_identifier
            url = hf_dataset_candidate_urls("hssd/hssd-models", row.file_identifier, proxy_prefix)
            future = executor.submit(
                direct_download_worker,
                url=url,
                destination=destination,
                expected_sha256=row.sha256,
                headers=headers,
                meter=meter,
            )
            future_to_row[future] = row

        for future in as_completed(future_to_row):
            row = future_to_row[future]
            try:
                status, resolved = future.result()
                tracker.record(
                    item_id=row.item_id,
                    source_key=source_key,
                    sha256=row.sha256,
                    status=status,
                    file_identifier=row.file_identifier,
                    local_path=relpath_to_output(output_root, resolved),
                )
            except Exception as exc:
                tracker.record(
                    item_id=row.item_id,
                    source_key=source_key,
                    sha256=row.sha256,
                    status=status_for_download_exception(exc),
                    file_identifier=row.file_identifier,
                    error=str(exc),
                )
            completed += 1
            meter.mark_item_complete()
            if completed % 50 == 0 or completed == len(pending):
                LOGGER.info("HSSD progress: %d/%d", completed, len(pending))
                meter.log_summary()
    meter.log_summary(force=True)


def extract_3d_future_instance(zip_handle: zipfile.ZipFile, instance: str, extract_root: Path) -> Path:
    members = [
        name
        for name in zip_handle.namelist()
        if name.startswith(f"{instance}/") and not name.endswith("/")
    ]
    if not members:
        raise FileNotFoundError(f"No members found for {instance}")
    zip_handle.extractall(extract_root, members=members)
    return extract_root / instance / "raw_model.obj"


def download_3d_future(
    rows: pd.DataFrame,
    output_root: Path,
    tracker: ProgressTracker,
    max_retries: int,
) -> None:
    source_key = "3d_future"
    archive = manual_archive_path(output_root, SOURCE_SPECS[source_key])
    if archive is None or not archive.exists():
        raise SourceUnavailable(
            f"Place `3D-FUTURE-model.zip` at `{archive}` before downloading 3D-FUTURE."
        )
    extract_root = source_root(output_root, source_key) / "raw"
    ensure_dir(extract_root)
    pending = [row for row in rows.itertuples(index=False) if should_process_row(row, tracker, max_retries)]
    if not pending:
        LOGGER.info("No pending 3D-FUTURE items.")
        return

    LOGGER.info("Extracting %d 3D-FUTURE items from %s", len(pending), archive)
    with zipfile.ZipFile(archive) as zip_handle:
        for index, row in enumerate(pending, start=1):
            image_path = extract_root / row.file_identifier / "image.jpg"
            target_model = extract_root / row.file_identifier / "raw_model.obj"
            try:
                if image_path.exists() and sha256_file(image_path) == row.sha256 and target_model.exists():
                    tracker.record(
                        item_id=row.item_id,
                        source_key=source_key,
                        sha256=row.sha256,
                        status="verified_existing",
                        file_identifier=row.file_identifier,
                        local_path=relpath_to_output(output_root, target_model),
                    )
                else:
                    if target_model.exists():
                        target_model.unlink()
                    if image_path.exists():
                        image_path.unlink()
                    extract_3d_future_instance(zip_handle, row.file_identifier, extract_root)
                    final_sha = sha256_file(image_path)
                    if final_sha != row.sha256:
                        tracker.record(
                            item_id=row.item_id,
                            source_key=source_key,
                            sha256=row.sha256,
                            status="modified",
                            file_identifier=row.file_identifier,
                            local_path=relpath_to_output(output_root, target_model),
                            error=f"expected image sha {row.sha256}, got {final_sha}",
                        )
                    else:
                        tracker.record(
                            item_id=row.item_id,
                            source_key=source_key,
                            sha256=row.sha256,
                            status="downloaded",
                            file_identifier=row.file_identifier,
                            local_path=relpath_to_output(output_root, target_model),
                        )
            except Exception as exc:
                tracker.record(
                    item_id=row.item_id,
                    source_key=source_key,
                    sha256=row.sha256,
                    status="failed",
                    file_identifier=row.file_identifier,
                    error=str(exc),
                )
            if index % 50 == 0 or index == len(pending):
                LOGGER.info("3D-FUTURE progress: %d/%d", index, len(pending))


def extract_toys4k_file(zip_handle: zipfile.ZipFile, filename: str, extract_root: Path) -> Path:
    member = f"toys4k_blend_files/{filename}"
    zip_handle.extract(member, extract_root)
    return extract_root / member


def download_toys4k(
    rows: pd.DataFrame,
    output_root: Path,
    tracker: ProgressTracker,
    max_retries: int,
) -> None:
    source_key = "toys4k"
    archive = manual_archive_path(output_root, SOURCE_SPECS[source_key])
    if archive is None or not archive.exists():
        raise SourceUnavailable(
            f"Place `toys4k_blend_files.zip` at `{archive}` before downloading Toys4k."
        )
    extract_root = source_root(output_root, source_key) / "raw"
    ensure_dir(extract_root)
    pending = [row for row in rows.itertuples(index=False) if should_process_row(row, tracker, max_retries)]
    if not pending:
        LOGGER.info("No pending Toys4k items.")
        return

    LOGGER.info("Extracting %d Toys4k items from %s", len(pending), archive)
    with zipfile.ZipFile(archive) as zip_handle:
        for index, row in enumerate(pending, start=1):
            final_path = extract_root / "toys4k_blend_files" / row.file_identifier
            try:
                if final_path.exists() and sha256_file(final_path) == row.sha256:
                    tracker.record(
                        item_id=row.item_id,
                        source_key=source_key,
                        sha256=row.sha256,
                        status="verified_existing",
                        file_identifier=row.file_identifier,
                        local_path=relpath_to_output(output_root, final_path),
                    )
                else:
                    final_path.unlink(missing_ok=True)
                    extract_toys4k_file(zip_handle, row.file_identifier, extract_root)
                    final_sha = sha256_file(final_path)
                    if final_sha != row.sha256:
                        tracker.record(
                            item_id=row.item_id,
                            source_key=source_key,
                            sha256=row.sha256,
                            status="modified",
                            file_identifier=row.file_identifier,
                            local_path=relpath_to_output(output_root, final_path),
                            error=f"expected {row.sha256}, got {final_sha}",
                        )
                    else:
                        tracker.record(
                            item_id=row.item_id,
                            source_key=source_key,
                            sha256=row.sha256,
                            status="downloaded",
                            file_identifier=row.file_identifier,
                            local_path=relpath_to_output(output_root, final_path),
                        )
            except Exception as exc:
                tracker.record(
                    item_id=row.item_id,
                    source_key=source_key,
                    sha256=row.sha256,
                    status="failed",
                    file_identifier=row.file_identifier,
                    error=str(exc),
                )
            if index % 50 == 0 or index == len(pending):
                LOGGER.info("Toys4k progress: %d/%d", index, len(pending))
def download_objaverse_source(
    rows: pd.DataFrame,
    output_root: Path,
    tracker: ProgressTracker,
    source_key: str,
    workers: int,
    max_retries: int,
    sketchfab_batch_size: int,
    proxy_prefix: Optional[str],
    speed_log_interval: float,
) -> None:
    spec = SOURCE_SPECS[source_key]
    if spec.kind == "objaverse_sketchfab":
        return download_objaverse_sketchfab(
            rows=rows,
            output_root=output_root,
            tracker=tracker,
            workers=workers,
            max_retries=max_retries,
            proxy_prefix=proxy_prefix,
            speed_log_interval=speed_log_interval,
        )
    if spec.kind != "objaverse_github":
        raise ValueError(f"Unsupported Objaverse kind: {spec.kind}")

    raw_dir = source_root(output_root, source_key) / "raw"
    ensure_dir(raw_dir)
    pending = [row for row in rows.itertuples(index=False) if should_process_row(row, tracker, max_retries)]
    if not pending:
        LOGGER.info("No pending %s items.", source_key)
        return

    LOGGER.info("Downloading %d ObjaverseXL GitHub items with %d workers", len(pending), workers)
    completed = 0
    meter = TransferMeter("ObjaverseXL GitHub", len(pending), speed_log_interval)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_row = {}
        for row in pending:
            try:
                relative_path = github_blob_relative_path(row.file_identifier)
                url = github_blob_to_raw_urls(row.file_identifier, proxy_prefix)
            except Exception as exc:
                tracker.record(
                    item_id=row.item_id,
                    source_key=source_key,
                    sha256=row.sha256,
                    status="failed",
                    file_identifier=row.file_identifier,
                    error=str(exc),
                )
                completed += 1
                meter.mark_item_complete()
                continue

            destination = raw_dir / relative_path
            future = executor.submit(
                direct_download_worker,
                url=url,
                destination=destination,
                expected_sha256=row.sha256,
                meter=meter,
            )
            future_to_row[future] = row

        for future in as_completed(future_to_row):
            row = future_to_row[future]
            try:
                status, destination = future.result()
                tracker.record(
                    item_id=row.item_id,
                    source_key=source_key,
                    sha256=row.sha256,
                    status=status,
                    file_identifier=row.file_identifier,
                    local_path=relpath_to_output(output_root, destination),
                )
            except Exception as exc:
                tracker.record(
                    item_id=row.item_id,
                    source_key=source_key,
                    sha256=row.sha256,
                    status=status_for_download_exception(exc),
                    file_identifier=row.file_identifier,
                    error=str(exc),
                )
            completed += 1
            meter.mark_item_complete()
            if completed % 50 == 0 or completed == len(pending):
                LOGGER.info("ObjaverseXL GitHub progress: %d/%d", completed, len(pending))
                meter.log_summary()
    meter.log_summary(force=True)


def process_source(
    source_key: str,
    rows: pd.DataFrame,
    output_root: Path,
    tracker: ProgressTracker,
    workers: int,
    max_retries: int,
    sketchfab_batch_size: int,
    hf_token: Optional[str],
    proxy_prefix: Optional[str],
    speed_log_interval: float,
) -> None:
    ready, reason = source_readiness(output_root, source_key)
    if not ready:
        raise SourceUnavailable(reason or f"{source_key} is not ready.")

    if source_key == "objaverse_xl_sketchfab":
        return download_objaverse_source(
            rows,
            output_root,
            tracker,
            source_key,
            workers,
            max_retries,
            sketchfab_batch_size,
            proxy_prefix,
            speed_log_interval,
        )
    if source_key == "objaverse_xl_github":
        return download_objaverse_source(
            rows,
            output_root,
            tracker,
            source_key,
            workers,
            max_retries,
            sketchfab_batch_size,
            proxy_prefix,
            speed_log_interval,
        )
    if source_key == "abo":
        return download_abo(rows, output_root, tracker, workers, max_retries, proxy_prefix, speed_log_interval)
    if source_key == "hssd":
        return download_hssd(
            rows,
            output_root,
            tracker,
            workers,
            max_retries,
            hf_token,
            proxy_prefix,
            speed_log_interval,
        )
    if source_key == "3d_future":
        return download_3d_future(rows, output_root, tracker, max_retries)
    if source_key == "toys4k":
        return download_toys4k(rows, output_root, tracker, max_retries)
    raise ValueError(f"Unsupported source: {source_key}")


def run_job(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    ensure_dir(output_root)
    detached = bool(os.environ.get("TRELLIS500K_DETACHED"))
    configure_logging(log_path(output_root), detached=detached)
    LOGGER.info("Starting TRELLIS-500K downloader job=%s output_root=%s", args.job, output_root)
    if args.web_proxy_prefix:
        LOGGER.info("Using web proxy prefix: %s", args.web_proxy_prefix)

    existing_alive, existing_pid = has_live_pid(pid_path(output_root))
    current_pid = os.getpid()
    if existing_alive and existing_pid != current_pid:
        LOGGER.error("Another downloader is already running for %s with pid %s", output_root, existing_pid)
        return 2
    write_pid_file(pid_path(output_root), current_pid)

    def cleanup_pid(*_unused) -> None:
        remove_pid_file(pid_path(output_root))

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(130))

    try:
        manifest = load_manifest(
            output_root=output_root,
            rebuild=args.rebuild_manifest,
            job=args.job,
            source_keys=args.source_keys,
            sample_size=args.sample_size,
            seed=args.seed,
            require_captions=not args.allow_missing_captions,
            refresh_metadata=args.refresh_metadata,
            available_only_for_minor=args.available_only_for_minor,
            proxy_prefix=args.web_proxy_prefix,
        )
        tracker = ProgressTracker(output_root)
        log_summary(manifest, tracker)
        if args.prepare_only:
            export_downloaded_pairs(output_root, manifest, tracker)
            cleanup_pid()
            return 0

        for source_key in args.source_keys:
            rows = manifest[manifest["source_key"] == source_key].copy()
            if rows.empty:
                continue
            LOGGER.info("Processing source %s (%d manifest rows)", source_key, len(rows))
            try:
                process_source(
                    source_key=source_key,
                    rows=rows,
                    output_root=output_root,
                    tracker=tracker,
                    workers=args.workers,
                    max_retries=args.max_retries,
                    sketchfab_batch_size=args.sketchfab_batch_size,
                    hf_token=args.hf_token,
                    proxy_prefix=args.web_proxy_prefix,
                    speed_log_interval=args.speed_log_interval,
                )
                clear_blocked_source(output_root, source_key)
            except SourceUnavailable as exc:
                save_blocked_source(output_root, source_key, str(exc))
                LOGGER.warning("Skipping %s for now: %s", source_key, exc)
            except Exception as exc:
                LOGGER.exception("Unhandled error while processing %s: %s", source_key, exc)
            log_summary(manifest, tracker)

        export_downloaded_pairs(output_root, manifest, tracker)
        LOGGER.info("TRELLIS-500K job finished for %s", output_root)
        return 0
    finally:
        cleanup_pid()


def launch_job(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    ensure_dir(output_root)
    alive, pid = has_live_pid(pid_path(output_root))
    if alive:
        print(f"Downloader is already running for {output_root} with pid {pid}.", flush=True)
        return 1

    ensure_dir(log_path(output_root).parent)
    command = [sys.executable, str(Path(__file__).resolve()), "run", args.job]
    command.extend(["--output-root", str(output_root)])
    command.extend(["--sample-size", str(args.sample_size)])
    command.extend(["--seed", str(args.seed)])
    command.extend(["--workers", str(args.workers)])
    command.extend(["--max-retries", str(args.max_retries)])
    command.extend(["--sketchfab-batch-size", str(args.sketchfab_batch_size)])
    command.extend(["--speed-log-interval", str(args.speed_log_interval)])
    if args.web_proxy_prefix is not None:
        command.extend(["--web-proxy-prefix", args.web_proxy_prefix])
    if args.sources is not None:
        command.extend(["--sources", args.sources])
    if args.include_toys4k:
        command.append("--include-toys4k")
    if args.allow_missing_captions:
        command.append("--allow-missing-captions")
    if args.rebuild_manifest:
        command.append("--rebuild-manifest")
    if args.refresh_metadata:
        command.append("--refresh-metadata")
    if args.prepare_only:
        command.append("--prepare-only")
    if not args.available_only_for_minor:
        command.append("--no-available-only-minor")

    env = os.environ.copy()
    env["TRELLIS500K_DETACHED"] = "1"
    with log_path(output_root).open("ab") as log_handle:
        proc = subprocess.Popen(
            command,
            cwd=str(repo_root()),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    write_pid_file(pid_path(output_root), proc.pid)
    print(f"Started pid={proc.pid}", flush=True)
    print(f"log={log_path(output_root)}", flush=True)
    print(f"manifest={manifest_path(output_root)}", flush=True)
    return 0


def status_job(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    manifest_file = manifest_path(output_root)
    if not manifest_file.exists():
        print(f"No manifest found at {manifest_file}")
        return 1
    manifest = pd.read_csv(manifest_file)
    tracker = ProgressTracker(output_root)
    blocked = read_json(blocked_sources_path(output_root), default={})
    alive, pid = has_live_pid(pid_path(output_root))

    print(f"output_root: {output_root}")
    print(f"running: {alive}")
    if pid is not None:
        print(f"pid: {pid}")
    print(f"log: {log_path(output_root)}")
    print(f"manifest_rows: {len(manifest)}")
    if blocked:
        print("blocked_sources:")
        for source_key, reason in blocked.items():
            print(f"  - {source_key}: {reason}")

    summary = summarize_progress(manifest, tracker)
    for source_key in manifest["source_key"].drop_duplicates().tolist():
        payload = summary.get(source_key, {"total": 0})
        print(
            f"{source_key}: total={payload.get('total', 0)} "
            f"downloaded={payload.get('downloaded', 0)} "
            f"verified={payload.get('verified_existing', 0)} "
            f"pending={payload.get('pending', 0)} "
            f"failed={payload.get('failed', 0)} "
            f"missing={payload.get('missing', 0)} "
            f"modified={payload.get('modified', 0)}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detached, resumable TRELLIS-500K downloader.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(subparser: argparse.ArgumentParser, include_prepare_only: bool) -> None:
        subparser.add_argument("job", choices=("minor", "entire"))
        subparser.add_argument("--output-root", type=Path, default=None)
        subparser.add_argument("--sources", type=str, default=None, help="Comma-separated source keys.")
        subparser.add_argument("--include-toys4k", action="store_true", help="Include the Toys4k eval set.")
        subparser.add_argument("--sample-size", type=int, default=4000, help="Only used for the `minor` job.")
        subparser.add_argument("--seed", type=int, default=42)
        subparser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
        subparser.add_argument("--max-retries", type=int, default=3)
        subparser.add_argument("--sketchfab-batch-size", type=int, default=64)
        subparser.add_argument("--speed-log-interval", type=float, default=30.0)
        subparser.add_argument("--hf-token", type=str, default=None)
        subparser.add_argument("--web-proxy-prefix", type=str, default=None)
        subparser.add_argument("--allow-missing-captions", action="store_true")
        subparser.add_argument("--rebuild-manifest", action="store_true")
        subparser.add_argument("--refresh-metadata", action="store_true")
        subparser.add_argument(
            "--no-available-only-minor",
            dest="available_only_for_minor",
            action="store_false",
            help="For the minor job, include sources even if they are not currently ready to download.",
        )
        subparser.set_defaults(available_only_for_minor=True)
        if include_prepare_only:
            subparser.add_argument("--prepare-only", action="store_true")

    launch_parser = subparsers.add_parser("launch", help="Run in the background and keep downloading after SSH disconnects.")
    add_common_arguments(launch_parser, include_prepare_only=True)

    run_parser = subparsers.add_parser("run", help="Run in the foreground.")
    add_common_arguments(run_parser, include_prepare_only=True)

    status_parser = subparsers.add_parser("status", help="Show manifest and download progress.")
    status_parser.add_argument("--output-root", type=Path, required=True)

    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if getattr(args, "job", None):
        if args.output_root is None:
            args.output_root = default_output_root(args.job)
        args.source_keys = parse_sources(args.job, args.sources, args.include_toys4k)
        if getattr(args, "hf_token", None) is None:
            args.hf_token = detected_hf_token()
        if getattr(args, "web_proxy_prefix", None) is None:
            args.web_proxy_prefix = DEFAULT_WEB_PROXY_PREFIX
        if args.job == "entire":
            args.sample_size = 0
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = normalize_args(parser.parse_args(argv))
    if args.command == "launch":
        return launch_job(args)
    if args.command == "run":
        return run_job(args)
    if args.command == "status":
        return status_job(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
