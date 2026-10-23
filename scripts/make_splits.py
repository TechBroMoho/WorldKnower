"""Create country_to_path.csv and train/val/test splits (Kartik's split code).

Run from the repo root:  python scripts/make_splits.py
Moved out of src/dataset.py so that importing the dataset module has no side effects.
Logic and seeds are unchanged, so it reproduces the same splits.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

dataset_path = "data/raw/compressed_dataset"

image_extensions = {".jpg", ".jpeg", ".png", ".webp"}


def main():
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
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=27, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.18, random_state=27, stratify=y_train_temp)

    os.makedirs('data/splits', exist_ok=True)
    for name, X_part, y_part in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        part = X_part.copy()
        part['country'] = y_part
        part.to_csv(f'data/splits/world_knower_{name}.csv', index=False)


if __name__ == "__main__":
    main()
