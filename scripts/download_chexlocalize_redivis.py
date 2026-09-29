#!/usr/bin/env python3
"""Download the pinned CheXlocalize v1.0 files through the Redivis REST API.

Only metadata and one bounded byte range are read in --probe mode. The full
mode preserves the official file-index paths and verifies each file's size and
publisher-provided MD5 digest. No third-party Python packages are required.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote


API_ROOT = "https://redivis.com/api/v1"
TABLE_REF = "aimi.chexlocalize:efx9:v1_0.chexlocalize:jcje"
TABLE_URL = f"{API_ROOT}/tables/{TABLE_REF}"
INDEX_URL = f"{TABLE_URL}/rows?format=jsonl"
MAX_INDEX_BYTES = 8 * 1024 * 1024
PROBE_BYTES = 4096
REQUIRED_FILES = {"gt_segmentations_val.json", "gt_segmentations_test.json"}


class DownloadError(Exception):
    pass


def check_curl() -> None:
    try:
        result = subprocess.run(
            ["curl", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=10, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise DownloadError("curl 8.4 or newer is required") from exc
    match = re.search(rb"^curl (\d+)\.(\d+)\.(\d+)", result.stdout)
    if result.returncode or not match or tuple(map(int, match.groups())) < (8, 4, 0):
        raise DownloadError("curl 8.4 or newer is required for bounded probes and safe redirects")


def curl_config(token: str) -> bytes:
    # The bearer value reaches curl over stdin, never as a process argument.
    if any(character in token for character in ('"', "\\", "\r", "\n")):
        raise DownloadError("Redivis token contains characters unsafe for curl's stdin config")
    return f'header = "Authorization: Bearer {token}"\n'.encode("utf-8")


def curl_base() -> list[str]:
    return [
        "curl", "--config", "-", "--silent", "--show-error", "--fail",
        "--location", "--max-redirs", "5", "--proto", "=https", "--proto-redir", "=https",
        "--connect-timeout", "15",
    ]


def curl_request(url: str, token: str, *, options: list[str], timeout: int = 90) -> bytes:
    result = subprocess.run(
        curl_base() + options + [url], input=curl_config(token),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False,
    )
    if result.returncode:
        raise DownloadError(f"Redivis request failed (curl exit {result.returncode})")
    return result.stdout


def read_limited(url: str, token: str, limit: int) -> bytes:
    payload = curl_request(url, token, options=["--max-time", "60", "--max-filesize", str(limit)])
    if len(payload) > limit:
        raise DownloadError("Redivis metadata exceeded the safe size limit")
    return payload


def load_token() -> str:
    for key in ("REDIVIS_API_TOKEN", "REDIVIS-DATA-API-KEY"):
        token = os.environ.get(key, "").strip()
        if token:
            return token

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.is_file():
        pattern = re.compile(r"^\s*(?:export\s+)?REDIVIS-DATA-API-KEY\s*=\s*(.*)$")
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            match = pattern.match(line)
            if match:
                token = match.group(1).strip()
                if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
                    token = token[1:-1]
                if token:
                    return token
    raise DownloadError(
        "Redivis token not found. Set REDIVIS_API_TOKEN, or provide "
        "REDIVIS-DATA-API-KEY in the repository's untracked .env file."
    )


@dataclass(frozen=True)
class FileRow:
    file_id: str
    name: str
    size: int
    md5: bytes


def parse_row(raw: dict) -> FileRow:
    file_id = raw.get("file_id")
    name = raw.get("file_name")
    size = raw.get("size")
    digest = raw.get("md5_hash")
    if not isinstance(file_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", file_id):
        raise DownloadError("File index contains an invalid file ID")
    if not isinstance(name, str) or not name or name.startswith("/") or "\\" in name or ":" in name:
        raise DownloadError("File index contains an unsafe file name")
    if any(part in ("", ".", "..") for part in name.split("/")):
        raise DownloadError("File index contains an unsafe path component")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise DownloadError("File index contains an invalid file size")
    if not isinstance(digest, str):
        raise DownloadError("File index lacks a file checksum")
    try:
        md5 = base64.b64decode(digest, validate=True)
    except ValueError as exc:
        raise DownloadError("File index contains an invalid MD5 checksum") from exc
    if len(md5) != 16:
        raise DownloadError("File index contains an invalid MD5 checksum")
    return FileRow(file_id, name, size, md5)


def table_row_count(token: str) -> int:
    data = json.loads(read_limited(TABLE_URL, token, 65536))
    count = data.get("numRows")
    if not isinstance(count, int) or count < 1:
        raise DownloadError("Redivis did not report a valid file-index row count")
    return count


def file_rows(token: str, *, probe: bool, expected_count: int) -> list[FileRow]:
    url = INDEX_URL + ("&maxResults=1" if probe else "")
    payload = read_limited(url, token, MAX_INDEX_BYTES)
    rows = [parse_row(json.loads(line)) for line in payload.splitlines() if line.strip()]
    if len(rows) != (1 if probe else expected_count):
        raise DownloadError(f"Redivis returned {len(rows)} index rows; expected {1 if probe else expected_count}")
    if len({row.name for row in rows}) != len(rows):
        raise DownloadError("Redivis file index contains duplicate paths")
    if not probe and not REQUIRED_FILES.issubset({row.name for row in rows}):
        raise DownloadError("Redivis index is missing required ground-truth segmentation files")
    return rows


def raw_url(row: FileRow) -> str:
    return f"{API_ROOT}/rawFiles/{quote(row.file_id, safe='._-')}"


def probe_file(row: FileRow, token: str) -> None:
    summary = curl_request(
        raw_url(row), token,
        options=["--range", f"0-{PROBE_BYTES - 1}", "--max-filesize", str(PROBE_BYTES * 2),
                 "--max-time", "45", "--output", os.devnull,
                 "--write-out", "%{http_code} %{size_download}"],
        timeout=60,
    ).decode("ascii", errors="replace")
    try:
        status_text, size_text = summary.split()
        status, size = int(status_text), int(size_text)
    except (ValueError, TypeError) as exc:
        raise DownloadError("Redivis probe did not report a valid HTTP status") from exc
    if status != 206 or not 0 < size <= PROBE_BYTES:
        raise DownloadError("Redivis did not return a bounded partial file response")
    print(f"PASS: Redivis file index is accessible; {row.name} returned HTTP 206 and {size} bytes (discarded).")


def within_root(root: Path, path: Path) -> bool:
    root_resolved = root.resolve()
    path_resolved = path.resolve()
    return os.path.commonpath((root_resolved, path_resolved)) == str(root_resolved)


def verify_file(path: Path, row: FileRow) -> bool:
    if not path.is_file() or path.stat().st_size != row.size:
        return False
    digest = hashlib.md5()  # noqa: S324 - compare against Redivis's published integrity digest
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest() == row.md5


def stream_to_part(row: FileRow, token: str, part: Path) -> None:
    for attempt in range(4):
        offset = part.stat().st_size if part.exists() else 0
        if offset > row.size:
            raise DownloadError(f"Oversized partial file: {row.name}")
        try:
            curl_request(
                raw_url(row), token,
                options=["--continue-at", "-", "--output", str(part),
                         "--max-filesize", str(row.size), "--max-time", "600"],
                timeout=630,
            )
            if part.stat().st_size == row.size:
                return
            if part.stat().st_size > row.size:
                raise DownloadError(f"Downloaded file exceeds the index size: {row.name}")
        except (DownloadError, subprocess.TimeoutExpired) as exc:
            if attempt == 3:
                raise DownloadError(f"Transfer failed for {row.name}: {type(exc).__name__}") from exc
        if attempt < 3:
            time.sleep(2 ** attempt)
    raise DownloadError(f"Incomplete transfer for {row.name}")


def download_one(row: FileRow, token: str, root: Path) -> str:
    destination = root.joinpath(*row.name.split("/"))
    part = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not within_root(root, destination) or not within_root(root, part):
        raise DownloadError(f"Destination escaped the chosen data directory: {row.name}")
    if destination.exists():
        if verify_file(destination, row):
            return "existing"
        raise DownloadError(f"Existing file failed size or checksum verification: {row.name}")
    if part.exists() and part.stat().st_size == row.size:
        if not verify_file(part, row):
            raise DownloadError(f"Partial file failed checksum verification: {row.name}")
    else:
        stream_to_part(row, token, part)
        if not verify_file(part, row):
            raise DownloadError(f"Downloaded file failed size or checksum verification: {row.name}")
    part.replace(destination)
    return "downloaded"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    root_arg = args.data_root
    if not root_arg.is_absolute() or root_arg.resolve() in (Path(root_arg.anchor), Path.home().resolve()):
        raise DownloadError("Choose an explicit absolute data directory, not a drive root or home directory")
    check_curl()
    token = load_token()
    count = table_row_count(token)
    rows = file_rows(token, probe=args.probe, expected_count=count)
    if args.probe:
        print(f"CheXlocalize Redivis v1.0 index reports {count} files; probing one file.")
        probe_file(rows[0], token)
        return 0

    root = root_arg / "_archives" / "chexlocalize" / "CheXlocalize"
    root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {count} official Redivis files to {root} (no CheXpert radiographs in this index).")
    completed = 0
    downloaded = 0
    errors = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download_one, row, token, root): row for row in rows}
        for future in as_completed(futures):
            completed += 1
            try:
                downloaded += future.result() == "downloaded"
            except Exception as exc:
                errors.append(str(exc))
            if completed % 100 == 0 or completed == count:
                print(f"CheXlocalize: {completed}/{count} checked; {downloaded} newly downloaded; {len(errors)} failed.", flush=True)
    if errors:
        for error in errors[:10]:
            print(f"ERROR: {error}", file=sys.stderr)
        raise DownloadError(f"{len(errors)} CheXlocalize files failed; rerun to resume verified partial transfers")
    print("CheXlocalize Redivis transfer complete; all files match the index sizes and MD5 digests.")
    print("Dataset-content and CheXpert image/label/mask alignment checks are still TODO after preparation.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (DownloadError, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
        print(f"CheXlocalize Redivis error: {error}", file=sys.stderr)
        sys.exit(1)
