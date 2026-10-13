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
  WorldKnowr.ipynb    # shared team notebook; must be added to this checkout
scripts/
  smoke_test.py       # temporary synthetic CPU integration test
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

`requirements.txt` includes the team's other dependencies; install it if you need their data/evaluation tools. On Colab, keep the existing CUDA-compatible torch/torchvision pair and install only missing supporting packages (`pandas`, `numpy`, `Pillow`). Verify that torch/torchvision import successfully and that PyTorch has the modern `torch.amp` APIs (PyTorch >= 2.4). Do not install the local CPU wheel command on the hosted runtime.

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

**Setup pending:** the shared `WorldKnowr.ipynb` is not present in this checkout. Add the original file at `notebooks/WorldKnowr.ipynb` before its setup/cache cells can be reviewed and the **Train ResNet18 Baseline** section integrated. The workflow below describes the intended integration; it does not claim the notebook has been modified or remotely verified.

1. Develop and run the CPU smoke test locally in VS Code. Keep the model, Dataset and training loop in `src/*.py`.
2. Review, commit and push your changes to [the team's GitHub repository](https://github.com/TechBroMoho/WorldKnower.git) on the intended team-approved branch. Local edits are not available to the remote runtime until pushed. Codex does not commit or push automatically.
3. Open `notebooks/WorldKnowr.ipynb` in VS Code. Choose **Select Kernel → Colab → WorldKnowr GPU T4** (or another hosted GPU server).
4. Run the shared environment setup: inspect `nvidia-smi`, verify `torch.cuda.is_available()` and print `torch.cuda.get_device_name(0)`. A GPU shown by `nvidia-smi` alone does not prove PyTorch can use it. Full training must refuse to run if CUDA is unavailable.
5. Mount Drive at `/content/drive` and complete authentication. If necessary use **Colab: Mount Google Drive to Server...**. Confirm the team's `WorldKnowerDrive` folder is accessible.
6. Clone the repository to `/content/WorldKnower` if absent. If present, check the branch and `git status --porcelain`; update a clean checkout with `git pull --ff-only`. Stop and resolve remote uncommitted changes or divergent branches manually. Never force-reset or discard edits. Use `%cd /content/WorldKnower` so `python -m src.train` runs from the actual repository root, independently of the notebook's location.
7. Use the shared dataset preparation from `/content/drive/MyDrive/WorldKnowerDrive/Data`, preserving its dataset ZIP cache. Verify extraction under `/content/data`, the expected images, and `country_to_path.csv`. Cache freshness/completeness and metadata copying on every loading branch need inspection once the notebook is supplied.
8. Configure the split and output paths below, validate the `path,country` split schema and resolved image files, then execute `src/train.py` with `--device cuda`. Do not generate real-data splits in this notebook.
9. Confirm `best_resnet18.pth`, `training_history.csv` and `experiment_log.csv` exist in the unique run directory on mounted Drive before disconnecting.

**After each source change:** save → test locally → commit/push the intended branch → rerun the notebook's safe Git update → launch a fresh training process. Check the remote commit and branch match the version you intend to train. For private repository access, authenticate through the team's approved method; do not embed credentials in the notebook.

Normal VS Code terminals run on your **local machine**. Cells connected to a hosted Colab kernel and an explicitly opened remote Colab terminal run on **Google's servers**. The directory name alone does not determine where execution happens. `/content` and its extracted dataset are temporary; mounted Drive outputs persist. Colab GPU allocation is not guaranteed and the code cannot force availability of a T4, L4, A100 or another GPU. See the extension's [official guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide) for hosted kernel selection and Drive mounting. Hosted execution and Drive access remain unverified locally.

## Switching to real data

The shared notebook has not been supplied, and the actual Drive dataset and split CSVs have not been accessed. These are proposed configurable paths, not verified dataset locations:

| Setting | Intended location |
| --- | --- |
| `DATASET_ROOT` | `/content/data/compressed_dataset` if split paths are relative to that folder |
| `TRAIN_CSV` | Teammate-provided `train.csv`; configure its actual Drive or extracted location |
| `VAL_CSV` | Teammate-provided `val.csv`; configure its actual Drive or extracted location |
| `CHECKPOINT_DIR` | `/content/drive/MyDrive/WorldKnowerDrive/results/<unique-run-id>/checkpoints` |
| `HISTORY_PATH` | `/content/drive/MyDrive/WorldKnowerDrive/results/<unique-run-id>/training_history.csv` |
| Experiment log | `/content/drive/MyDrive/WorldKnowerDrive/results/<unique-run-id>/experiment_log.csv` |

Choose a fresh timestamp/UUID run ID every time you launch training, and refuse an already-existing run directory. This isolates your outputs from earlier experiments and other teammates. The checkpoint file is `CHECKPOINT_DIR/best_resnet18.pth`.

Inspect `country_to_path.csv` separately; its name does not establish its schema, and it must not be substituted for `train.csv`/`val.csv` without verification. Kartik and Sri provide the real splits. Before training, use `CountryDataset(TRAIN_CSV, DATASET_ROOT)` and construct validation with its `country_to_idx`; verify resolved image paths remain inside the selected extracted dataset. For `France/image001.jpg`, a root of `/content/data/compressed_dataset` resolves the image there. For `compressed_dataset/France/image001.jpg`, the root should instead be `/content/data`. No split generation or source rewrite is needed when compatible CSVs arrive.

Keep `test.csv` for James's evaluation. Training never reads it. Country count is inferred from training. The intended single hyperparameter cell uses 10 epochs, batch size 32, learning rate 0.0001, seed 42, pretrained weights/augmentation/AMP enabled, and CUDA. Adam and CrossEntropyLoss remain defined in `src/train.py`. Reduce batch size for CUDA out-of-memory errors. Worker count is 0 by default in the CLI for Windows compatibility; 2 is a reasonable initial Colab setting.

## Training CLI and artifacts

From the project root (full training should run on the hosted runtime):

```bash
python -m src.train --train-csv data/splits/train.csv --val-csv data/splits/val.csv --dataset-root data/processed --device cuda
python -m src.train --help
```

`python src/train.py` is supported too. Defaults: pretrained ResNet18, full fine-tuning, Adam, CrossEntropyLoss, learning rate 0.0001, batch size 32, 10 epochs, augmentation enabled, seed 42. `--device auto` selects CUDA when available; `--device cuda` requires it; `--device cpu` forces CPU. Local tests use CPU explicitly. Flags include `--no-pretrained`, `--no-augmentation`, `--no-amp`, `--num-workers`, `--seed`, `--checkpoint-dir`, and `--history-path`.

Best validation accuracy saves `results/checkpoints/best_resnet18.pth`; history goes to `results/training_history.csv`, with `epoch,train_loss,val_accuracy`. Loss is sample-weighted and accuracy is a **fraction from 0 to 1**. The first epoch is saved even if accuracy is zero; ties retain the earlier best. These output paths are configurable and each run overwrites its own history/best file. Use distinct output directories for separate experiments.

`--experiment-log results/experiment_log.csv` appends a completed run using the existing schema without overwriting records. For hosted training pass the unique run's Drive log path instead. `val_f1` is blank because James owns F1 evaluation. `--author` and `--notes` can be supplied. The project's existing log remains unchanged until you opt into logging a real run.

Seeds cover Python, NumPy, PyTorch, CUDA and loader workers. CUDA defaults to cuDNN benchmarking for fixed-size input performance; `--deterministic` disables benchmarking and requests deterministic algorithms. For strict CUDA runs, set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before launching Python. Reproducibility is not guaranteed across hardware/library versions. Checkpoints include optimizer/scaler state and settings, but this baseline does not implement a resume command or save a complete random-generator snapshot.

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
