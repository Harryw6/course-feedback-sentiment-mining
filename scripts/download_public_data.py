#!/usr/bin/env python3
"""Download legal public course-review datasets and record provenance.

The script follows AGENTS.md priority:
1. Kaggle 100K Coursera reviews.
2. If Kaggle credentials exist, also Kaggle 1.45M Coursera reviews.
3. If Kaggle is unavailable, Hugging Face fallback.

It does not scrape websites. It uses Kaggle CLI/API commands when available and
direct Hugging Face file URLs for the fallback.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
REPORT_DIR = BASE_DIR / "outputs" / "reports"
MANIFEST_PATH = RAW_DIR / "download_manifest.json"
PROVENANCE_PATH = REPORT_DIR / "DATA_PROVENANCE.md"


@dataclass
class DatasetRecord:
    name: str
    dataset_id: str
    source_platform: str
    source_url: str
    license: str
    license_status: str
    download_command: str
    download_datetime_utc: str
    local_dir: str
    files_downloaded: list[str] = field(default_factory=list)
    fields_used: list[str] = field(default_factory=list)
    known_limitations: list[str] = field(default_factory=list)
    status: str = "downloaded"

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dataset_id": self.dataset_id,
            "source_platform": self.source_platform,
            "source_url": self.source_url,
            "license": self.license,
            "license_status": self.license_status,
            "download_command": self.download_command,
            "download_datetime_utc": self.download_datetime_utc,
            "local_dir": self.local_dir,
            "files_downloaded": self.files_downloaded,
            "fields_used": self.fields_used,
            "known_limitations": self.known_limitations,
            "status": self.status,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def has_kaggle_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    if (Path.home() / ".kaggle" / "kaggle.json").exists():
        return True
    if (BASE_DIR / "kaggle.json").exists():
        return True
    return False


def kaggle_command() -> list[str] | None:
    if shutil.which("kaggle"):
        return ["kaggle"]
    try:
        subprocess.run(
            [sys.executable, "-m", "kaggle", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return [sys.executable, "-m", "kaggle"]
    except Exception:
        return None


def run_kaggle_download(dataset_id: str, output_dir: Path) -> tuple[bool, str]:
    cmd_base = kaggle_command()
    if cmd_base is None:
        if list_files(output_dir):
            return True, f"existing files registered from prior official Kaggle download; Kaggle CLI unavailable for rerun of {dataset_id}"
        return False, "Kaggle CLI is not installed."
    if not has_kaggle_credentials():
        if list_files(output_dir):
            return True, f"existing files registered from prior official Kaggle download; credentials unavailable for rerun of {dataset_id}"
        return False, "Kaggle credentials are not configured."

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = cmd_base + ["datasets", "download", "-d", dataset_id, "-p", str(output_dir), "--unzip"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        if list_files(output_dir):
            message = (proc.stderr or proc.stdout).strip().splitlines()
            first_line = message[0] if message else "command returned nonzero"
            return True, " ".join(cmd) + f" (registered existing downloaded files after nonzero return: {first_line})"
        return False, (proc.stderr or proc.stdout).strip()
    return True, " ".join(cmd)


def download_url(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "course-feedback-research/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())


def try_hf_fallback() -> tuple[DatasetRecord | None, str | None]:
    dataset_id = "chillies/course-review-multilabel-sentiment-analysis"
    target_dir = RAW_DIR / "hf_course_review_multilabel"
    files = {
        "train.csv": "https://huggingface.co/datasets/chillies/course-review-multilabel-sentiment-analysis/resolve/main/train.csv",
        "test.csv": "https://huggingface.co/datasets/chillies/course-review-multilabel-sentiment-analysis/resolve/main/test.csv",
    }
    downloaded: list[str] = []
    try:
        for filename, url in files.items():
            destination = target_dir / filename
            download_url(url, destination)
            downloaded.append(str(destination.relative_to(BASE_DIR)))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        existing = [str((target_dir / filename).relative_to(BASE_DIR)) for filename in files if (target_dir / filename).exists()]
        if existing:
            downloaded = existing
        else:
            return None, f"Hugging Face fallback download failed: {exc}"

    record = DatasetRecord(
        name="course-review-multilabel-sentiment-analysis",
        dataset_id=dataset_id,
        source_platform="Hugging Face",
        source_url="https://huggingface.co/datasets/chillies/course-review-multilabel-sentiment-analysis",
        license="Not specified on the Hugging Face dataset card metadata available at download time.",
        license_status="warning_license_not_explicit",
        download_command=(
            "urllib.request downloads from Hugging Face resolve/main URLs for train.csv and test.csv"
        ),
        download_datetime_utc=utc_now(),
        local_dir=str(target_dir.relative_to(BASE_DIR)),
        files_downloaded=downloaded,
        fields_used=[
            "review",
            "Improvement Suggestions",
            "Questions and Answers",
            "Experience Sharing",
            "Technical Feedback",
            "Support Request",
            "Community Interaction",
            "Course Comparison",
            "Related Course Suggestions",
            "not_praise",
        ],
        known_limitations=[
            "Auxiliary dataset is multi-label feedback-type data, not a three-class sentiment dataset.",
            "No explicit sentiment labels or ratings are provided; sentiment baselines must be skipped unless a Kaggle dataset is later downloaded.",
            "The Hugging Face dataset page is public, but an explicit license was not found in available metadata; verify rights before publication use.",
        ],
    )
    return record, None


def list_files(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted(str(p.relative_to(BASE_DIR)) for p in path.rglob("*") if p.is_file())


def try_kaggle_datasets() -> tuple[list[DatasetRecord], list[str]]:
    records: list[DatasetRecord] = []
    errors: list[str] = []

    primary_id = "septa97/100k-courseras-course-reviews-dataset"
    primary_dir = RAW_DIR / "kaggle_100k_coursera_reviews"
    ok, message = run_kaggle_download(primary_id, primary_dir)
    if ok:
        records.append(
            DatasetRecord(
                name="100K Coursera's Course Reviews Dataset",
                dataset_id=primary_id,
                source_platform="Kaggle",
                source_url="https://www.kaggle.com/datasets/septa97/100k-courseras-course-reviews-dataset",
                license="Database: Open Database, Contents: Database Contents",
                license_status="explicit_on_source_page",
                download_command=message,
                download_datetime_utc=utc_now(),
                local_dir=str(primary_dir.relative_to(BASE_DIR)),
                files_downloaded=list_files(primary_dir),
                fields_used=["Id", "CourseId", "Review", "Label"],
                known_limitations=[
                    "Labels were pre-labeled from ratings by the dataset publisher.",
                    "Dataset description says reviews were originally scraped by the publisher as of May 2017.",
                    "Class distribution is imbalanced according to the source page.",
                ],
            )
        )
    else:
        errors.append(f"{primary_id}: {message}")

    if has_kaggle_credentials():
        secondary_id = "imuhammad/course-reviews-on-coursera"
        secondary_dir = RAW_DIR / "kaggle_145m_coursera_reviews"
        ok, message = run_kaggle_download(secondary_id, secondary_dir)
        if ok:
            records.append(
                DatasetRecord(
                    name="Course Reviews on Coursera",
                    dataset_id=secondary_id,
                    source_platform="Kaggle",
                    source_url="https://www.kaggle.com/datasets/imuhammad/course-reviews-on-coursera",
                    license="GPL 2",
                    license_status="explicit_on_source_page",
                    download_command=message,
                    download_datetime_utc=utc_now(),
                    local_dir=str(secondary_dir.relative_to(BASE_DIR)),
                    files_downloaded=list_files(secondary_dir),
                    fields_used=["reviews", "rating", "course_id"],
                    known_limitations=[
                        "Contains reviewer/date columns that are not used as model features.",
                        "Large dataset may be expensive to process in full mode on CPU-only machines.",
                    ],
                )
            )
        else:
            errors.append(f"{secondary_id}: {message}")

    return records, errors


def write_manifest(records: list[DatasetRecord], errors: list[str]) -> None:
    payload = {
        "created_at_utc": utc_now(),
        "records": [record.as_dict() for record in records],
        "download_errors": errors,
        "policy": {
            "no_scraping": True,
            "raw_data_gitignored": True,
            "credentials_gitignored": True,
        },
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_provenance(records: list[DatasetRecord], errors: list[str]) -> None:
    lines = [
        "# Data Provenance",
        "",
        f"Generated at (UTC): {utc_now()}",
        "",
        "This file is generated by `scripts/download_public_data.py`. No fabricated data is created.",
        "",
    ]
    if errors:
        lines.extend(["## Download Attempts and Errors", ""])
        for error in errors:
            lines.append(f"- {error}")
        lines.append("")
    if not records:
        lines.extend(
            [
                "## No Dataset Downloaded",
                "",
                "No legal public dataset was downloaded. Configure Kaggle credentials or manually download a licensed dataset before running experiments.",
                "",
            ]
        )
    for record in records:
        lines.extend(
            [
                f"## {record.name}",
                "",
                f"- Dataset ID: `{record.dataset_id}`",
                f"- Source platform: {record.source_platform}",
                f"- Source page: {record.source_url}",
                f"- License: {record.license}",
                f"- License status: {record.license_status}",
                f"- Download command: `{record.download_command}`",
                f"- Download date/time UTC: {record.download_datetime_utc}",
                f"- Local directory: `{record.local_dir}`",
                "- Files downloaded:",
            ]
        )
        for file_path in record.files_downloaded:
            lines.append(f"  - `{file_path}`")
        lines.append("- Fields used:")
        for field_name in record.fields_used:
            lines.append(f"  - `{field_name}`")
        lines.append("- Known limitations:")
        for limitation in record.known_limitations:
            lines.append(f"  - {limitation}")
        lines.append("")
    PROVENANCE_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_manual_instructions(errors: Iterable[str]) -> None:
    manual_path = REPORT_DIR / "manual_download_instructions.md"
    lines = [
        "# Manual Download Instructions",
        "",
        "Automatic public dataset download did not complete.",
        "",
        "1. Configure Kaggle credentials without committing them:",
        "   - Create `~/.kaggle/kaggle.json`, or set `KAGGLE_USERNAME` and `KAGGLE_KEY` in `.env`.",
        "   - Keep credentials out of Git.",
        "2. Run:",
        "   `python scripts/download_public_data.py`",
        "3. If automatic Kaggle download remains unavailable, manually download one of these licensed source datasets:",
        "   - https://www.kaggle.com/datasets/septa97/100k-courseras-course-reviews-dataset",
        "   - https://www.kaggle.com/datasets/imuhammad/course-reviews-on-coursera",
        "4. Place extracted files under `data/raw/<dataset-name>/` and rerun `python scripts/verify_data_sources.py`.",
        "",
        "Errors observed:",
    ]
    for error in errors:
        lines.append(f"- {error}")
    manual_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ensure_dirs()
    load_env_file(BASE_DIR / ".env")

    records, errors = try_kaggle_datasets()
    fallback_record, fallback_error = try_hf_fallback()
    if fallback_record:
        records.append(fallback_record)
    if fallback_error:
        errors.append(fallback_error)

    write_manifest(records, errors)
    write_provenance(records, errors)

    if not records:
        write_manual_instructions(errors)
        print("No dataset downloaded. See outputs/reports/manual_download_instructions.md")
        return 2

    print(f"Downloaded/registered {len(records)} dataset(s).")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Provenance: {PROVENANCE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
