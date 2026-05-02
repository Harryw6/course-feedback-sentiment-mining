#!/usr/bin/env python3
"""Verify downloaded dataset provenance metadata before experiments run."""

from __future__ import annotations

import json
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = BASE_DIR / "data" / "raw" / "download_manifest.json"
PROVENANCE_PATH = BASE_DIR / "outputs" / "reports" / "DATA_PROVENANCE.md"


REQUIRED_FIELDS = {
    "name",
    "dataset_id",
    "source_platform",
    "source_url",
    "license",
    "license_status",
    "download_command",
    "download_datetime_utc",
    "files_downloaded",
    "fields_used",
    "known_limitations",
}


def main() -> int:
    if not MANIFEST_PATH.exists():
        print(f"Missing manifest: {MANIFEST_PATH}", file=sys.stderr)
        return 2
    if not PROVENANCE_PATH.exists():
        print(f"Missing provenance report: {PROVENANCE_PATH}", file=sys.stderr)
        return 2

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    if not records:
        print("Manifest contains no downloaded dataset records.", file=sys.stderr)
        return 2

    failures: list[str] = []
    warnings: list[str] = []
    for idx, record in enumerate(records, start=1):
        missing = sorted(field for field in REQUIRED_FIELDS if not record.get(field))
        if missing:
            failures.append(f"record {idx} missing required metadata: {', '.join(missing)}")
        local_dir = BASE_DIR / str(record.get("local_dir", ""))
        if not local_dir.exists():
            failures.append(f"record {idx} local directory does not exist: {local_dir}")
        for file_path in record.get("files_downloaded", []):
            if not (BASE_DIR / file_path).exists():
                failures.append(f"record {idx} missing downloaded file: {file_path}")
        if "not specified" in str(record.get("license", "")).lower():
            warnings.append(
                f"record {idx} has no explicit source license; use with publication caution: {record.get('dataset_id')}"
            )

    if failures:
        print("Data source verification failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"Verified {len(records)} dataset record(s).")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
