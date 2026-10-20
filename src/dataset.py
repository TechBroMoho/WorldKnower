import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

"""Creating country to image_path csv and train/val/test splits"""

dataset_path = "data/raw/compressed_dataset"

image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

country_to_image_data = {
    "country": [],
    "image_path": []
}

rng = np.random.default_rng(42)

for country in sorted(os.listdir(dataset_path)):
    country_path = os.path.join(dataset_path, country)

    if not os.path.isdir(country_path):
        continue

    country_image_arr = np.array([
        os.path.relpath(os.path.join(root, file), country_path)
        for root, dirs, files in os.walk(country_path)
        for file in files
        if os.path.splitext(file)[1].lower() in image_extensions
    ])
    country_sample = rng.choice(country_image_arr, size=min(250, country_image_arr.size), replace=False)

    for image_file in country_sample:
        country_to_image_data['country'].append(country)
        image_path = os.path.join(country_path, image_file)
        country_to_image_data['image_path'].append(image_path)

country_to_image_df = pd.DataFrame(country_to_image_data)
country_to_image_df.to_csv('data/raw/country_to_path.csv', index=False)

X = country_to_image_df[['image_path']]
y = country_to_image_df['country']
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size = 0.15, random_state = 27, stratify = y)
X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size = 0.18, random_state = 27, stratify = y_train_temp)

country_to_image_train_df = X_train.copy()
country_to_image_train_df['country'] = y_train
country_to_image_train_df.to_csv('data/splits/world_knower_train.csv', index=False)

country_to_image_val_df = X_val.copy()
country_to_image_val_df['country'] = y_val
country_to_image_val_df.to_csv('data/splits/world_knower_val.csv', index=False)

country_to_image_test_df = X_test.copy()
country_to_image_test_df['country'] = y_test
country_to_image_test_df.to_csv('data/splits/world_knower_test.csv', index=False)


"""CSV image datasets, ImageNet transforms, and ordinary PyTorch DataLoaders."""

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
    """Build labels on training; pass its mapping to validation and test."""

    def __init__(self, csv_path, root_dir=None, country_to_idx=None, transform=None):
        csv_path = Path(csv_path)
        root = Path(root_dir) if root_dir is not None else csv_path.parent
        frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        if "image_path" in frame.columns and "path" not in frame.columns:
          frame = frame.rename(columns={"image_path": "path"})
        if not {"path", "country"}.issubset(frame.columns):
            raise ValueError(f"{csv_path} must contain columns 'path' and 'country'")
        if frame.empty or frame[["path", "country"]].eq("").any().any():
            raise ValueError(f"{csv_path} must contain samples with nonempty paths and countries")
        self.country_to_idx = (dict(country_to_idx) if country_to_idx is not None
                               else {name: i for i, name in enumerate(sorted(frame.country.unique()))})
        unknown = set(frame.country) - self.country_to_idx.keys()
        if unknown:
            raise ValueError(f"{csv_path}: unknown countries {sorted(unknown)}; use training mapping")
        self.countries = sorted(self.country_to_idx, key=self.country_to_idx.get)
        self.transform = transform if transform is not None else build_transforms()
        self.image_paths = [root / Path(path) for path in frame.path]
        self.labels = [self.country_to_idx[country] for country in frame.country]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        if not path.is_file():
            raise FileNotFoundError(f"Missing image: {path}; check dataset root")
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.labels[index]


def create_loader(dataset, batch_size=32, num_workers=0, training=False, device="cpu", seed=42):
    return DataLoader(dataset, batch_size=batch_size, shuffle=training,
                      num_workers=num_workers, pin_memory=torch.device(device).type == "cuda",
                      generator=torch.Generator().manual_seed(seed))


def create_dataloaders(train_csv, val_csv, root_dir=None, batch_size=32,
                       num_workers=0, augmentation=True, device="cpu", seed=42):
    train = CountryDataset(train_csv, root_dir, transform=build_transforms(True, augmentation))
    val = CountryDataset(val_csv, root_dir, country_to_idx=train.country_to_idx,
                         transform=build_transforms())
    return (create_loader(train, batch_size, num_workers, True, device, seed),
            create_loader(val, batch_size, num_workers, False, device, seed))
