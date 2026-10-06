# WorldKnower

Predict which country a street-level image was taken in. We fine-tune a pretrained CNN (e.g. ResNet via torchvision) on GeoGuessr-style images and treat it as a classification problem with one class per country.

Inspired by [callaunchpad/GeoKnowr](https://github.com/callaunchpad/GeoKnowr). That project predicts U.S. states; we extend the idea to countries worldwide.

## Project structure

```
data/
  raw/          # original downloaded images (not in git)
  processed/    # resized / cleaned images (not in git)
  splits/       # train/val/test CSVs (not in git)
notebooks/      # exploration and quick experiments
src/
  dataset.py    # PyTorch Dataset + transforms
  model.py      # CNN definition (pretrained backbone + classifier head)
  train.py      # training loop
  evaluate.py   # accuracy, F1, confusion matrix
  geo_utils.py  # lat/lon -> country lookup (GeoPandas), helpers
results/
  checkpoints/  # saved model weights (not in git)
  figures/      # plots, confusion matrices
  experiment_log.csv  # one row per training run
```

## Setup

Requires Python 3.11 or newer.

```bash
git clone https://github.com/techbromoho/WorldKnower.git
cd WorldKnower
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Images and checkpoints are **not** stored in git because they are too large. Download them here:

**Google Drive:** https://drive.google.com/drive/folders/1ft7XsTZN0EtjHrBf-4OX56wlKJv9-i3v

Unzip the files into `data/raw/` so that you have `data/raw/<image files>` plus the metadata CSV.

## Training and evaluation

Run these from the repo root with the virtual environment activated:

```bash
python src/train.py
python src/evaluate.py
```

After every training run, add a row to `results/experiment_log.csv`.

## Team workflow

Five of us are pushing this week, so follow these rules to avoid merge conflicts:

1. **Never commit directly to `main`.** Create a branch for each task, for example `git checkout -b yourname/dataset-loader`.
2. **Open a pull request** for anything that touches shared files (`src/`, `requirements.txt`, `README.md`, `.gitignore`). Get at least one teammate to review it before merging.
3. **Pull often:** run `git pull origin main` before you start working, then merge `main` into your branch before you open a PR.
4. **Don't commit data or weights.** Upload them to the shared Drive instead.
5. If you add a library, pin its version in `requirements.txt` and mention it in your PR.
