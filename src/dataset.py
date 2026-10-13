"""CSV image datasets, shared label mappings, and reproducible loaders."""

from pathlib import Path, PureWindowsPath
import random
import shutil
import hashlib

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def build_transforms(training=False, augmentation=True):
    """Only training may use random augmentation; geographic directions stay intact."""
    steps = (
        [transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
         transforms.RandomRotation(10), transforms.ColorJitter(brightness=0.15)]
        if training and augmentation else [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    )
    return transforms.Compose(steps + [transforms.ToTensor(),
                                       transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class CountryDataset(Dataset):
    """Relative paths resolve under root_dir (CSV parent when root_dir is omitted).

    Build country_to_idx on training only, then pass it to every other split.
    Use forward-slash relative paths in shared CSVs for Windows/Colab portability.
    """

    def __init__(self, csv_path, root_dir=None, country_to_idx=None, transform=None):
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.root_dir = (Path(root_dir).expanduser().resolve()
                         if root_dir is not None else self.csv_path.parent)
        try:
            frame = pd.read_csv(self.csv_path, dtype=str, keep_default_na=False)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise ValueError(f"Cannot read split CSV {self.csv_path}: {exc}") from exc
        if not {"path", "country"}.issubset(frame.columns):
            raise ValueError(f"{self.csv_path} must contain columns 'path' and 'country'")
        if frame.empty:
            raise ValueError(f"{self.csv_path} contains no samples")
        for column in ("path", "country"):
            if frame[column].str.strip().eq("").any():
                raise ValueError(f"{self.csv_path}: empty {column} at one or more rows")
        self.country_to_idx = (dict(country_to_idx) if country_to_idx is not None
                               else {name: i for i, name in enumerate(sorted(frame.country.unique()))})
        indices = list(self.country_to_idx.values())
        if (not indices or any(type(i) is not int for i in indices)
                or sorted(indices) != list(range(len(indices)))):
            raise ValueError("country_to_idx must contain unique consecutive integer indices from zero")
        unknown = set(frame.country) - self.country_to_idx.keys()
        if unknown:
            raise ValueError(f"{self.csv_path}: unknown countries {sorted(unknown)}; use training mapping")
        self.countries = sorted(self.country_to_idx, key=self.country_to_idx.get)
        self.transform = transform if transform is not None else build_transforms()
        self.frame = frame
        self.image_paths = [self.resolve_image_path(p) for p in frame.path]
        self.labels = [self.country_to_idx[country] for country in frame.country]
        for row, path in enumerate(self.image_paths, start=2):
            if not path.is_file():
                raise FileNotFoundError(f"{self.csv_path}, row {row}: missing image {path}; check root_dir")

    def resolve_image_path(self, value):
        # Accept relative backslashes too, but never reinterpret a Windows drive on Linux.
        path = Path(value.replace("\\", "/")).expanduser()
        if PureWindowsPath(value).is_absolute() and not path.is_absolute():
            raise ValueError(f"Windows absolute path {value!r} cannot be used here; use relative paths")
        return path if path.is_absolute() else self.root_dir / path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        try:
            with Image.open(path) as image:
                tensor = self.transform(image.convert("RGB"))
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Cannot decode image {path} from {self.csv_path}: {exc}") from exc
        return tensor, self.labels[index]


def seed_worker(worker_id):
    """Top-level function so Windows spawn workers can pickle it."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def create_loader(dataset, batch_size=32, num_workers=0, training=False, device="cpu", seed=42):
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    return DataLoader(dataset, batch_size=batch_size, shuffle=training,
                      num_workers=num_workers, pin_memory=torch.device(device).type == "cuda",
                      worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))


def create_dataloaders(train_csv, val_csv, root_dir=None, batch_size=32,
                       num_workers=0, augmentation=True, device="cpu", seed=42):
    train = CountryDataset(train_csv, root_dir, transform=build_transforms(True, augmentation))
    val = CountryDataset(val_csv, root_dir, country_to_idx=train.country_to_idx,
                         transform=build_transforms())
    return (create_loader(train, batch_size, num_workers, True, device, seed),
            create_loader(val, batch_size, num_workers, False, device, seed))


def stage_selected_data(train_csv, val_csv, root_dir, destination):
    """Copy only referenced train/val images; rewrite CSVs for temporary local storage.

    Returns (staged_root, staged_train_csv, staged_val_csv). No test data is copied.
    """
    destination = Path(destination).resolve()
    train = CountryDataset(train_csv, root_dir)
    val = CountryDataset(val_csv, root_dir, country_to_idx=train.country_to_idx)
    destination.mkdir(parents=True, exist_ok=True)
    for name, dataset in (("train", train), ("val", val)):
        records = []
        for source, country in zip(dataset.image_paths, dataset.frame.country):
            digest = hashlib.sha256(str(source.resolve()).encode()).hexdigest()
            relative = Path("images") / (digest + source.suffix.lower())
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            # Refresh images when staging again after dataset changes.
            shutil.copy2(source, target)
            records.append({"path": relative.as_posix(), "country": country})
        pd.DataFrame(records).to_csv(destination / f"{name}.csv", index=False)
    return destination, destination / "train.csv", destination / "val.csv"
