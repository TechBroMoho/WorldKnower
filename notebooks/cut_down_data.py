import os
import shutil
from collections import Counter

dataset_path = "data/raw/compressed_dataset"

image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

country_counts = Counter()
removed_countries = []

for country in os.listdir(dataset_path):
    country_path = os.path.join(dataset_path, country)

    if not os.path.isdir(country_path):
        continue

    count = sum(
        1
        for root, dirs, files in os.walk(country_path)
        for file in files
        if os.path.splitext(file)[1].lower() in image_extensions
    )

    if count >= 200:
        country_counts[country] = count
    else:
        shutil.rmtree(country_path)
        removed_countries.append(country)

for country, count in sorted(
    country_counts.items(),
    key=lambda item: item[1],
    reverse=True
):
    print(f"{country}: {count} files")

print(f"\nTotal countries with >=200 files: {len(country_counts)}")
print(f"Removed {len(removed_countries)} countries with <200 files: {', '.join(sorted(removed_countries))}")