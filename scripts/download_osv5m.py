"""Download the OSV-5M images chosen by select_osv5m.py, without downloading whole zips.

Each image is read straight out of its remote zip with one HTTP range request and saved as
IMAGE_ROOT/<Country>/osv5m/<id>.jpg, so expand_train.py picks it up as part of that country.
Existing files are skipped, so an interrupted run can simply be restarted. Set HF_TOKEN in the
environment for higher Hugging Face rate limits.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import struct
import threading
import time
import zipfile
import zlib

import pandas as pd
import requests

from select_osv5m import REPO, HTTPFile

LOCAL_HEADER = struct.Struct("<4s5H3I2H")  # Fixed 30-byte part of a zip local file header.
_local = threading.local()


def session():
    """One requests.Session per worker thread (sessions are not thread-safe)."""
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
        if os.environ.get("HF_TOKEN"):
            # requests drops this header when Hugging Face redirects to its file CDN.
            _local.session.headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    return _local.session


def resolve(zip_path):
    """Return (direct CDN URL, {member: ZipInfo}) for one remote zip."""
    with zipfile.ZipFile(HTTPFile(f"{REPO}/{zip_path}", session())) as archive:
        return archive.fp.url, {info.filename: info for info in archive.infolist()}


def fetch(url, info, attempts=5):
    """Read one member's bytes with a single range request, and check its CRC."""
    # The local header's name/extra lengths can differ from the central directory's, so
    # over-fetch a little and parse the real lengths from the local header itself.
    end = info.header_offset + LOCAL_HEADER.size + len(info.filename) + 1024 + info.compress_size
    for attempt in range(attempts):
        try:
            response = session().get(url, headers={"Range": f"bytes={info.header_offset}-{end}"}, timeout=60)
            if response.status_code == 403:
                raise PermissionError("CDN link expired")
            response.raise_for_status()
            blob = response.content
            fields = LOCAL_HEADER.unpack_from(blob)
            if fields[0] != b"PK\x03\x04":
                raise ValueError(f"Bad local header for {info.filename}")
            start = LOCAL_HEADER.size + fields[9] + fields[10]
            data = blob[start:start + info.compress_size]
            if info.compress_type == zipfile.ZIP_DEFLATED:
                data = zlib.decompress(data, -15)
            elif info.compress_type != zipfile.ZIP_STORED:
                raise ValueError(f"Unsupported compression {info.compress_type} for {info.filename}")
            if zlib.crc32(data) != info.CRC:
                raise ValueError(f"CRC mismatch for {info.filename}")
            return data
        except PermissionError:
            raise
        except (requests.RequestException, ValueError, zlib.error):
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def save(data, path):
    temporary = path.with_suffix(".part")
    temporary.write_bytes(data)
    temporary.replace(path)  # Atomic, so a killed run never leaves a truncated .jpg behind.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-csv", type=Path, default=Path("data/raw/osv5m/osv5m_selection_1000.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("data/raw/compressed_dataset"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-images", type=int, help="Download only this many (for a quick test)")
    args = parser.parse_args()
    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN not set: downloading unauthenticated (lower rate limits).", flush=True)

    selection = pd.read_csv(args.selection_csv, dtype=str)
    selection["target"] = [args.image_root / country / "osv5m" / f"{image_id}.jpg"
                           for country, image_id in zip(selection.country_name, selection.id)]
    for folder in {target.parent for target in selection.target}:
        folder.mkdir(parents=True, exist_ok=True)
    for leftover in args.image_root.glob("*/osv5m/*.part"):
        leftover.unlink()
    todo = selection[[not target.exists() for target in selection.target]]
    if args.max_images is not None:
        todo = todo.head(args.max_images)
    print(f"{len(selection) - len(todo)} already present, {len(todo)} to download "
          f"from {todo.zip.nunique()} zips", flush=True)

    done, failed, started = 0, [], time.time()
    for zip_number, (zip_path, group) in enumerate(todo.groupby("zip"), 1):
        url, members = resolve(zip_path)
        print(f"[{zip_number}/{todo.zip.nunique()}] {zip_path}: {len(group)} images", flush=True)
        pending = list(group.itertuples(index=False))
        while pending:
            retry = []
            with ThreadPoolExecutor(args.workers) as pool:
                futures = {pool.submit(fetch, url, members[row.member]): row for row in pending}
                for future in as_completed(futures):
                    row = futures[future]
                    try:
                        save(future.result(), row.target)
                        done += 1
                        if done % 500 == 0:
                            print(f"  {done}/{len(todo)} downloaded ({time.time() - started:.0f}s)", flush=True)
                    except PermissionError:
                        retry.append(row)
                    except Exception as error:  # Keep going; report failures at the end.
                        failed.append((row.id, row.country_name, str(error)))
            if retry:  # The signed CDN link expired mid-zip; get a fresh one and finish the rest.
                url, members = resolve(zip_path)
            pending = retry

    present = selection[[target.exists() for target in selection.target]]
    expected = selection.groupby("country_name").size()
    counts = present.groupby("country_name").size().reindex(expected.index, fill_value=0)
    report = pd.DataFrame({"expected": expected, "on_disk": counts})
    print(report.to_string())
    print(f"\nDownloaded {done} this run in {time.time() - started:.0f}s | "
          f"{len(present)}/{len(selection)} selected images on disk | {len(failed)} failed")
    for image_id, country, error in failed[:20]:
        print(f"  FAILED {country} {image_id}: {error}")
    if failed:
        print("Rerun the same command to retry the failures.")


if __name__ == "__main__":
    main()
