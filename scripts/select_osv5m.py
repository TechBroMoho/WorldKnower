"""Choose OSV-5M images to top up each country's training pool, and the zips that hold them.

Reads only metadata and the zips' file listings (via HTTP range requests); no images are
downloaded. Countries are topped up to --limit using OSV-5M's hand-checked test split first,
then its train split, taking at most --max-per-sequence images per Mapillary sequence.
Zips are chosen greedily so as few as possible need downloading.
"""

import argparse
import io
from pathlib import Path
import zipfile

import pandas as pd
import requests

REPO = "https://huggingface.co/datasets/osv5m/osv5m/resolve/main"
TREE = "https://huggingface.co/api/datasets/osv5m/osv5m/tree/main/images"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ISO = {
    "Argentina": "AR", "Australia": "AU", "Austria": "AT", "Belgium": "BE", "Brazil": "BR",
    "Bulgaria": "BG", "Canada": "CA", "Chile": "CL", "Colombia": "CO", "Czechia": "CZ",
    "Finland": "FI", "France": "FR", "Germany": "DE", "Greece": "GR", "Indonesia": "ID",
    "Ireland": "IE", "Israel": "IL", "Italy": "IT", "Japan": "JP", "Malaysia": "MY",
    "Mexico": "MX", "Netherlands": "NL", "New Zealand": "NZ", "Norway": "NO", "Peru": "PE",
    "Philippines": "PH", "Poland": "PL", "Portugal": "PT", "Romania": "RO", "Russia": "RU",
    "Singapore": "SG", "South Africa": "ZA", "South Korea": "KR", "Spain": "ES", "Sweden": "SE",
    "Taiwan": "TW", "Thailand": "TH", "Turkey": "TR", "United Kingdom": "GB", "United States": "US",
}


class HTTPFile(io.RawIOBase):
    """Read-only, seekable view of a remote file, so zipfile can list it without downloading."""

    def __init__(self, url, session):
        response = session.head(url, allow_redirects=True)
        response.raise_for_status()
        self.url, self.size, self.pos, self.session = response.url, int(response.headers["Content-Length"]), 0, session

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        response = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        response.raise_for_status()
        self.pos += len(response.content)
        return response.content


def geoguessr_pool(image_root, held_out_csvs):
    """Images per country available for training: everything on disk not in val/test."""
    held_out = set()
    for path in held_out_csvs:
        frame = pd.read_csv(path)
        column = "path" if "path" in frame.columns else "image_path"
        held_out |= {p.replace("\\", "/").split("compressed_dataset/")[-1] for p in frame[column]}
    pool = {}
    for country in ISO:
        folder = image_root / country
        files = [p.relative_to(image_root).as_posix() for p in folder.rglob("*")
                 if p.is_file() and p.suffix.lower() in EXTENSIONS] if folder.is_dir() else []
        pool[country] = sum(f not in held_out for f in files)
    return pool


def load_metadata(cache_dir, codes):
    cache = cache_dir / "osv5m_metadata_filtered.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype=str)
    frames = []
    for split in ("test", "train"):
        print(f"Streaming {split}.csv metadata...", flush=True)
        # requests rather than a pandas URL: it ships its own CA certificates.
        with requests.get(f"{REPO}/{split}.csv", stream=True) as response:
            response.raise_for_status()
            response.raw.decode_content = True
            for chunk in pd.read_csv(response.raw, usecols=["id", "country", "sequence"],
                                     dtype=str, keep_default_na=False, chunksize=500_000):
                frames.append(chunk[chunk.country.isin(codes)].assign(split=split))
    metadata = pd.concat(frames, ignore_index=True)
    metadata.to_csv(cache, index=False)
    return metadata


def load_zip_index(cache_dir, wanted_ids):
    """Map image id -> zip path for every OSV-5M zip, by reading each zip's file listing."""
    cache = cache_dir / "osv5m_zip_index.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype=str)
    session = requests.Session()
    rows = []
    for split in ("test", "train"):
        zips = sorted(f["path"] for f in session.get(f"{TREE}/{split}").json() if f["path"].endswith(".zip"))
        for number, zip_path in enumerate(zips, 1):
            print(f"Listing {zip_path} ({number}/{len(zips)})", flush=True)
            with zipfile.ZipFile(HTTPFile(f"{REPO}/{zip_path}", session)) as archive:
                for member in archive.namelist():
                    image_id = Path(member).stem
                    if image_id in wanted_ids:
                        rows.append((image_id, zip_path, member))
    index = pd.DataFrame(rows, columns=["id", "zip", "member"])
    index.to_csv(cache, index=False)
    return index


def select(candidates, need, max_per_sequence, seed):
    """Greedy: test zips before train zips, each time taking the zip that fills the most need."""
    remaining = dict(need)
    used_sequences = {}
    chosen = []
    candidates = candidates.sample(frac=1, random_state=seed)  # Random order within each zip.
    by_zip = {name: group for name, group in candidates.groupby("zip")}
    for cap in range(1, max_per_sequence + 1):
        for split in ("test", "train"):
            pending = {z for z in by_zip if z.startswith(f"images/{split}/")}
            while pending and any(remaining.values()):
                def take(zip_path):
                    taken, counts, sequences = [], {}, {}
                    for row in by_zip[zip_path].itertuples(index=False):
                        key = (row.country, row.sequence)
                        if counts.get(row.country, 0) >= remaining.get(row.country, 0):
                            continue
                        if used_sequences.get(key, 0) + sequences.get(key, 0) >= cap:
                            continue
                        taken.append(row)
                        counts[row.country] = counts.get(row.country, 0) + 1
                        sequences[key] = sequences.get(key, 0) + 1
                    return taken
                gains = {z: take(z) for z in pending}
                best = max(gains, key=lambda z: (len(gains[z]), z))
                if not gains[best]:
                    break
                for row in gains[best]:
                    key = (row.country, row.sequence)
                    used_sequences[key] = used_sequences.get(key, 0) + 1
                    remaining[row.country] -= 1
                    chosen.append(row)
                by_zip[best] = by_zip[best][~by_zip[best].id.isin({r.id for r in gains[best]})]
                pending.discard(best)
    return pd.DataFrame(chosen), remaining


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=Path("data/raw/compressed_dataset"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/splits/world_knower_val.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/splits/world_knower_test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/osv5m"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-per-sequence", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pool = geoguessr_pool(args.image_root, [args.val_csv, args.test_csv])
    need = {ISO[c]: max(0, args.limit - n) for c, n in pool.items() if n < args.limit}
    metadata = load_metadata(args.output_dir, set(need))
    metadata = metadata[metadata.country.isin(need)]
    index = load_zip_index(args.output_dir, set(metadata.id))
    candidates = metadata.merge(index, on="id")
    chosen, remaining = select(candidates, need, args.max_per_sequence, args.seed)

    name = {code: country for country, code in ISO.items()}
    chosen.insert(1, "country_name", chosen.country.map(name))
    chosen.to_csv(args.output_dir / f"osv5m_selection_{args.limit}.csv", index=False)
    summary = pd.DataFrame([dict(country=name[code], geoguessr_pool=pool[name[code]], need=n,
                                 from_osv_test=int(((chosen.country == code) & (chosen.split == "test")).sum()),
                                 from_osv_train=int(((chosen.country == code) & (chosen.split == "train")).sum()),
                                 short=remaining[code]) for code, n in need.items()])
    summary["final_total"] = (summary.geoguessr_pool.clip(upper=args.limit) + summary.from_osv_test
                              + summary.from_osv_train)
    summary.to_csv(args.output_dir / f"osv5m_selection_{args.limit}_summary.csv", index=False)
    zips = chosen.groupby("zip").size().rename("images").reset_index()
    zips.to_csv(args.output_dir / f"osv5m_selection_{args.limit}_zips.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nSelected {len(chosen)} images from {len(zips)} zips (~{len(zips) * 2.5:.0f} GB to download):")
    print(zips.to_string(index=False))


if __name__ == "__main__":
    main()
