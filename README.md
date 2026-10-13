# WorldKnower

Predict which country a street-level image was taken in. The baseline fine-tunes the full ImageNet-pretrained torchvision ResNet18, with one output class per country.

Inspired by [callaunchpad/GeoKnowr](https://github.com/callaunchpad/GeoKnowr). That project predicts U.S. states; we extend the idea to countries worldwide.

## Project structure

```
data/
  raw/          # original downloaded images (not in git)
  processed/    # resized / cleaned images (not in git)
  splits/       # train/val/test CSVs (not in git)
notebooks/
  train_resnet18_colab.ipynb  # hosted GPU execution interface
scripts/
  smoke_test.py       # temporary synthetic CPU integration test
  package_colab.py    # bundle current source for upload
src/
  dataset.py    # PyTorch Dataset + transforms
  model.py      # ResNet18 factory and checkpoint reconstruction
  train.py      # training loop
  evaluate.py   # James's evaluation assignment (currently empty)
  geo_utils.py  # teammate's geographic helpers (currently empty)
results/
  checkpoints/  # saved model weights (not in git)
  figures/      # plots, confusion matrices
  experiment_log.csv  # one row per training run
```

## Local development and CPU smoke test

Requires Python 3.11 or newer. Work in the existing checkout. To install the baseline locally without setting up CUDA:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS instead: source .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.2.6 pandas==2.2.3 Pillow==11.3.0
python scripts/smoke_test.py
```

The smoke test explicitly uses CPU, disables pretrained weights, limits CPU threads, and runs one epoch on nine tiny random RGB images (six train, three validation). It also checks non-RGB conversion, label reuse, errors, worker spawning, selected-data staging, backpropagation, history, and checkpoint reload. All images/results are created in a separate temporary directory and removed afterward. Accuracy on random images has no scientific meaning. It never touches project checkpoints or the existing experiment log.

`requirements.txt` includes the team's other dependencies; install it if you need their data/evaluation tools. On Colab, keep the existing CUDA-compatible torch/torchvision pair; the notebook installs only missing supporting packages. Do not install the local CPU wheel command on the hosted runtime.

## Data and relative paths

Images and checkpoints are **not** stored in git because they are too large. Download them here:

**[Shared Google Drive](https://drive.google.com/drive/folders/1ft7XsTZN0EtjHrBf-4OX56wlKJv9-i3v)**

Kartik and Sri prepare images and `data/splits/train.csv`, `val.csv`, and `test.csv`. The baseline does not create real-data splits or balance countries. Each split must have these columns:

```csv
path,country
images/France/image001.jpg,France
images/Japan/image002.jpg,Japan
```

`CountryDataset(csv_path, root_dir=...)` resolves a relative image path against `root_dir`, independently of where the CSV lives. If no root is supplied, it uses the CSV's parent directory. For the example above, set the root to the folder **containing `images/`**. CLI `--dataset-root` defaults to the working directory. Absolute image paths work on the OS where they exist; use forward-slash relative paths for the same CSV to work on Windows and Colab. Relative backslashes are accepted too. Windows absolute paths cannot be reused on Linux; rewrite those CSV paths as relative paths first.

Country indices come from sorted unique training country names. Validation/test must pass that saved mapping; a split with a previously unseen country fails clearly. Images are decoded with PIL and converted to RGB. Validation/test resize to 224×224 and use ImageNet normalization. Optional training augmentation uses moderate crops (80–100% area), ±10° rotations and slight brightness variation, with no horizontal flips.

## Google Colab through VS Code

1. Open `notebooks/train_resnet18_colab.ipynb`. Use **Select Kernel → Colab**, then choose your existing **WorldKnowr GPU T4** server (or another hosted GPU server). Run section 1: it must report CUDA and the cloud GPU's name. A local kernel or CPU runtime is rejected.
2. In your **local terminal**, run `python scripts/package_colab.py`. Right-click `dist/worldknower_source.zip` in VS Code Explorer → **Upload to Colab**, choosing the intended server if prompted. Use the Colab Contents view to find its uploaded path. Section 2 defaults to `/content/worldknower_source.zip`; adjust `SOURCE_ZIP` if needed. Extraction puts the Python project in `/content/WorldKnower`. Direct source-folder uploads are also supported by setting `SOURCE_ZIP = None` and the correct `PROJECT_SOURCE`.
3. Run section 3 to verify dependencies without replacing Colab's torch/torchvision stack.
4. Run section 4 to mount Drive and authenticate. If mounting fails in the notebook, use **Colab: Mount Google Drive to Server...**, then confirm `/content/drive/MyDrive` exists. Set the dataset root and CSV paths to your team's uploaded data. Set output paths on mounted Drive; the default creates a separate timestamped run directory.
5. Run sections 5–6 to validate files and set hyperparameters. Optionally enable `STAGE_IMAGES` to copy only referenced train/validation images to `/content/worldknower_staged` for faster reads.
6. Run section 7. It launches the uploaded `src/train.py` with **`--device cuda`** in a fresh subprocess. It uses the selected cloud GPU, AMP, pinned memory and nonblocking transfers. No model or training loop is duplicated in the notebook.
7. Run sections 8–9 to display history and verify checkpoint/history/manifest on Drive. Check those files in Drive before disconnecting. The uploaded source, staged data and weight-download cache in `/content` are temporary and may disappear; Drive outputs persist.

**Synchronize after every Codex/local edit:** save the Python files → rerun `python scripts/package_colab.py` locally → upload/replace the ZIP on the same server → rerun notebook section 2 and sections 3–9. Uploads are explicit; editing local files does not update remote copies. Section 2 prints source hashes and training records them in `run_manifest.json`. The fresh training subprocess avoids stale Python imports. Rerun section 4 for a new output run directory; rerunning section 7 alone overwrites that run's checkpoint/history.

**Website fallback:** upload this notebook at [Google Colab](https://colab.research.google.com/), select a GPU through Runtime settings, upload the same source ZIP with the Files sidebar (or `google.colab.files.upload()` on the website), and run the same sections. The ZIP includes only Python source, this notebook, README and requirements, excluding images, environments, weights, and `src/model.ipynb`.

The extension's [official user guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide) documents kernel selection, explicit uploads and Drive mounting. Its [known issues](https://github.com/googlecolab/colab-vscode/wiki/Known-Issues-and-Workarounds) explain why website-specific file upload APIs may not work in VS Code. Upload/Drive authentication and hosted GPU execution require manual execution in your extension; local smoke testing does not verify them. Colab GPU allocation depends on availability; this code does not request a particular GPU model.

## Switching to real data

Change only the notebook configuration: `DATASET_ROOT`, `TRAIN_CSV`, `VAL_CSV`, `CHECKPOINT_DIR`, and `HISTORY_PATH` (plus `PROJECT_SOURCE`/`SOURCE_ZIP` for the upload location). Default CSVs are under `MyDrive/WorldKnower/data/splits/`; the example dataset root is `MyDrive/WorldKnower/data/processed/`. Adjust the root to your actual CSV path convention. Outputs default to a run directory under `MyDrive/WorldKnower/results/`.

Keep `test.csv` for James's evaluation. Training and staging never read it. The number of countries is inferred from training, so no source changes are needed for a new country set. Set epochs, batch size, learning rate and augmentation in section 6. Reduce batch size for CUDA out-of-memory errors. Worker count is 0 by default in the CLI for Windows compatibility, and 2 in the Colab notebook.

## Training CLI and artifacts

From the project root (full training should run on the hosted runtime):

```bash
python -m src.train --train-csv data/splits/train.csv --val-csv data/splits/val.csv --dataset-root data/processed --device cuda
python -m src.train --help
```

`python src/train.py` is supported too. Defaults: pretrained ResNet18, full fine-tuning, Adam, CrossEntropyLoss, learning rate 0.0001, batch size 32, 10 epochs, augmentation enabled, seed 42. `--device auto` selects CUDA when available; `--device cuda` requires it; `--device cpu` forces CPU. Local tests use CPU explicitly. Flags include `--no-pretrained`, `--no-augmentation`, `--no-amp`, `--num-workers`, `--seed`, `--checkpoint-dir`, and `--history-path`.

Best validation accuracy saves `results/checkpoints/best_resnet18.pth`; history goes to `results/training_history.csv`, with `epoch,train_loss,val_accuracy`. Loss is sample-weighted and accuracy is a **fraction from 0 to 1**. The first epoch is saved even if accuracy is zero; ties retain the earlier best. These output paths are configurable and each run overwrites its own history/best file. Use distinct output directories for separate experiments.

`--experiment-log results/experiment_log.csv` appends a completed run using the existing schema without overwriting records. The Colab notebook enables this on Drive. `val_f1` is blank because James owns F1 evaluation. `--author` and `--notes` can be supplied. The project's existing log remains unchanged until you opt into logging a real run.

Seeds cover Python, NumPy, PyTorch, CUDA and loader workers. CUDA defaults to cuDNN benchmarking for fixed-size input performance; `--deterministic` disables benchmarking and requests deterministic algorithms. For strict CUDA runs, set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before launching Python (the notebook does this automatically). Reproducibility is not guaranteed across hardware/library versions. Checkpoints include optimizer/scaler state and settings, but this baseline does not implement a resume command or save a complete random-generator snapshot.

## Evaluation compatibility

`src/evaluate.py` is currently empty and was preserved for James. Checkpoints expose `architecture` (`resnet18`), `model_state_dict`, `country_to_idx`, `countries` in index order, `num_classes`, `epoch`, `best_val_accuracy`, optimizer/scaler state, configuration, image size and normalization statistics. Reload without downloading pretrained weights:

```python
import torch
from src.model import load_checkpoint
from src.dataset import CountryDataset, create_loader, build_transforms

model, checkpoint = load_checkpoint("results/checkpoints/best_resnet18.pth", device="cpu")
countries = checkpoint["countries"]
test = CountryDataset("data/splits/test.csv", root_dir="data/processed",
                      country_to_idx=checkpoint["country_to_idx"],
                      transform=build_transforms())
loader = create_loader(test, batch_size=32, training=False, device="cpu")
images, labels = next(iter(loader))
with torch.inference_mode():
    indices = model(images).argmax(dim=1).tolist()
predicted_countries = [countries[index] for index in indices]
```

The helper loads with `weights_only=True`, validates class metadata, builds `build_model(checkpoint['num_classes'], pretrained=False)`, restores weights, and returns an eval-mode model plus metadata. James can use these interfaces for his accuracy/F1/confusion matrix/report implementation. On CUDA evaluation, pass `device="cuda"` to the helper/loader and move images to that device before inference.

## Team workflow

Five of us are pushing this week, so follow these rules to avoid merge conflicts:

1. **Never commit directly to `main`.** Create a branch for each task, for example `git checkout -b yourname/dataset-loader`.
2. **Open a pull request** for anything that touches shared files (`src/`, `requirements.txt`, `README.md`, `.gitignore`). Get at least one teammate to review it before merging.
3. **Pull often:** run `git pull origin main` before you start working, then merge `main` into your branch before you open a PR.
4. **Don't commit data or weights.** Upload them to the shared Drive instead.
5. If you add a library, pin its version in `requirements.txt` and mention it in your PR.
