"""Add training images without re-splitting the baseline. Standard library only.

image-root must contain the full source dataset's COUNTRY/image.jpg folders.
The output CSV uses paths relative to image-root. Existing outputs are refused.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import zipfile

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def image_key(value, country):
    parts = PurePosixPath(value.replace("\\", "/")).parts
    if "compressed_dataset" in parts:
        parts = parts[parts.index("compressed_dataset") + 1:]
    if len(parts) < 2 or parts[0] != country or ".." in parts:
        raise ValueError(f"Unrecognized image path for {country}: {value}")
    return PurePosixPath(*parts).as_posix()


def read_split(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        column = "path" if "path" in columns else "image_path"
        if "country" not in columns or column not in columns:
            raise ValueError(f"{path}: expected path (or image_path) and country")
        rows = [(image_key(row[column], row["country"]), row["country"])
                for row in reader]
    if not rows or len({key for key, _ in rows}) != len(rows):
        raise ValueError(f"{path}: empty split or duplicate paths")
    return rows


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def expand(args):
    inputs = [args.train_csv, args.val_csv, args.test_csv]
    before = {str(path): sha256(path) for path in inputs}
    splits = [read_split(path) for path in inputs]
    train, val, test = splits
    keys = [{key for key, _ in rows} for rows in splits]
    if any(keys[a] & keys[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Original splits overlap; resolve with your lead before proceeding")
    countries = sorted({country for _, country in train})
    if any(country not in countries for _, country in val + test):
        raise ValueError("A held-out split contains a country absent from training")
    for key, _ in train + val + test:
        if not (args.image_root / key).is_file():
            raise FileNotFoundError(f"Source image missing: {args.image_root / key}")

    output = args.output
    counts_path = output.with_name(output.stem + "_counts.csv")
    manifest_path = output.with_name(output.stem + "_manifest.json")
    outputs = [output, counts_path, manifest_path]
    if args.zip_output:
        outputs.append(args.zip_output)
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError("Output paths must be different")
    for path in outputs:
        if path.exists() or path.resolve() in {p.resolve() for p in inputs}:
            raise FileExistsError(f"Refusing to overwrite {path}")
    if args.limit < 1:
        raise ValueError("limit must be positive")

    blocked = set.union(*keys)
    rng = random.Random(args.seed)
    expanded = list(train)  # Keep every baseline training row, in its original order.
    counts = []
    for country in countries:
        existing = sum(label == country for _, label in train)
        if existing > args.limit:
            raise ValueError(f"{country} already has {existing} training images, above limit")
        folder = args.image_root / country
        available = sorted(p.relative_to(args.image_root).as_posix()
                           for p in folder.rglob("*")
                           if p.is_file() and p.suffix.lower() in EXTENSIONS)
        candidates = [key for key in available if key not in blocked]
        new = rng.sample(candidates, min(args.limit - existing, len(candidates)))
        expanded.extend((key, country) for key in new)
        counts.append((country, existing, len(new), existing + len(new)))

    expanded_keys = {key for key, _ in expanded}
    if len(expanded_keys) != len(expanded) or not keys[0] <= expanded_keys:
        raise AssertionError("Training preservation or uniqueness failed")
    if expanded_keys & (keys[1] | keys[2]):
        raise AssertionError("Training overlaps a held-out split")
    if len(expanded) <= len(train):
        raise ValueError("No training images added; image-root may contain only the old subset")
    if any(sha256(path) != before[str(path)] for path in inputs):
        raise RuntimeError("An original split changed during preparation")

    write_csv(output, ["path", "country"], expanded)
    write_csv(counts_path, ["country", "old_train", "added", "new_train"], counts)
    manifest = dict(seed=args.seed, limit=args.limit, countries=countries,
                    original_split_sha256=before, train_csv_sha256=sha256(output),
                    original_train=len(train), expanded_train=len(expanded),
                    validation=len(val), test_membership_only=len(test),
                    image_root=str(args.image_root), overlap_check="normalized file paths",
                    zip_output=str(args.zip_output) if args.zip_output else None)
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    if args.zip_output:
        args.zip_output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(args.zip_output, "x", compression=zipfile.ZIP_STORED) as archive:
                for key in sorted(expanded_keys | keys[1] | keys[2]):
                    archive.write(args.image_root / key, f"data/raw/compressed_dataset/{key}")
                for path in inputs + [output, counts_path, manifest_path]:
                    archive.write(path, f"data/splits/{path.name}")
        except Exception:
            args.zip_output.unlink(missing_ok=True)
            raise
    if any(sha256(path) != before[str(path)] for path in inputs):
        raise RuntimeError("An original split changed during packaging")
    print(f"Countries: {len(countries)} | training: {len(train)} -> {len(expanded)}")
    print(f"Validation: {len(val)} | test membership: {len(test)} | new overlap: 0")
    for row in counts:
        print(f"{row[0]}: {row[1]} + {row[2]} = {row[3]}")
    print(f"Saved {output}, {counts_path}, {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--val-csv", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path,
                        help="Read for exclusion only; no test evaluation")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zip-output", type=Path,
                        help="New archive including expanded training and unchanged held-out images")
    expand(parser.parse_args())


if __name__ == "__main__":
    main()
