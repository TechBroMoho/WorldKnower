import os
import numpy as np
import pandas as pd

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
country_to_image_df.to_csv('data/country_to_path.csv', index=False)

