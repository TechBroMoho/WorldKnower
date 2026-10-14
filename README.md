# WorldKnower

Predict the country where a photograph was taken by fine-tuning an ImageNet-pretrained ResNet18.
Inspired by [GeoKnowr](https://github.com/callaunchpad/GeoKnowr).

## Baseline files

- `src/dataset.py`: CSV Dataset, RGB conversion, transforms and DataLoaders.
- `src/model.py`: ResNet18 factory and checkpoint-loading helper.
- `src/train.py`: Adam/CrossEntropy training, validation, best checkpoint and history CSV.
- `scripts/smoke_test.py`: small CPU test using temporary synthetic images.
- `src/evaluate.py` and `src/geo_utils.py`: teammates' files, left unchanged.
- `data/{raw,processed,splits}`: datasets and splits, excluded from Git.
- `results/checkpoints`: saved weights, excluded from Git.

The team's online **WorldKnowr.ipynb** handles Colab setup, Drive mounting, repository updates and dataset extraction. It is not maintained in this checkout.

## Local setup and testing

Use Python 3.11 or newer. Install CPU-only dependencies in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.2.6 pandas==2.2.3 Pillow==11.3.0
python scripts/smoke_test.py
python src/train.py --help
```

On Linux/macOS, activate with `source .venv/bin/activate`. The existing `requirements.txt` also includes teammates' dependencies.

The smoke test forces CPU, disables pretrained downloads, and runs one short epoch on six training and three validation images. It checks RGB conversion, transforms, shared labels, batches, output dimensions, backpropagation and checkpoint reload. Temporary images/results are removed. Random-image accuracy is not meaningful.

## Data format

Kartik and Sri supply train/validation/test CSVs with `path,country` columns:

```csv
path,country
France/image001.jpg,France
Japan/image002.jpg,Japan
```

Set `--dataset-root` to the folder containing those relative paths. For example, if images are under `/content/data/compressed_dataset/France/`, use `/content/data/compressed_dataset` as the root. If CSV paths already begin with `compressed_dataset/`, use `/content/data` instead. Absolute paths work on the OS where they exist; use forward-slash relative paths to share CSVs between Windows and Colab. With no root supplied to `CountryDataset`, paths resolve against the CSV's parent; the training CLI defaults to the working directory.

Labels are sorted once from training and reused for validation/test. Unknown countries raise an error. `country_to_path.csv` is not assumed to be a split CSV. The real data and splits have not been accessed or validated here; their paths must be confirmed when teammates provide them. Training never reads the test split or generates real-data splits.

## Train through the existing Colab notebook

1. Develop and test locally in VS Code, then review, commit and push the intended branch to [the team repository](https://github.com/TechBroMoho/WorldKnower).
2. Open the existing online `WorldKnowr.ipynb` through your normal VS Code workflow and connect it to a hosted GPU using the official Colab extension.
3. Run its setup, safely update the cloned repository, and load/extract the dataset from Drive. Local edits reach Colab only after you push them and update the remote checkout. Do not force-reset a checkout with edits.
4. Manually add this cell after setup. Replace all four uppercase path placeholders:

```python
%cd /content/WorldKnower
!python -u -m src.train \
    --device cuda \
    --train-csv "TRAIN_CSV_PATH" \
    --val-csv "VAL_CSV_PATH" \
    --dataset-root "IMAGE_ROOT_PATH" \
    --checkpoint-dir "/content/drive/MyDrive/WorldKnowerDrive/results/EXPERIMENT_FOLDER/checkpoints" \
    --history-path "/content/drive/MyDrive/WorldKnowerDrive/results/EXPERIMENT_FOLDER/training_history.csv"
```

Use a different `EXPERIMENT_FOLDER` for each experiment you want to keep. The same paths are overwritten on a later run. Mounted Drive results persist; `/content/data` and the cloned repository are temporary. Normal VS Code terminals run locally; hosted notebook cells and an explicitly remote Colab terminal run on Google's servers. GPU availability is not guaranteed. `--device cuda` fails if PyTorch cannot access CUDA.

Defaults are pretrained ResNet18, full fine-tuning, 10 epochs, batch size 32, learning rate 0.0001, Adam, CrossEntropyLoss and seed 42. Augmentation uses moderate crops, small rotations and brightness changes without horizontal flips; validation/test only resize to 224×224 and apply ImageNet normalization. CUDA mixed precision is enabled by default; `--no-amp` disables it. Other overrides include `--epochs`, `--batch-size`, `--lr`, `--num-workers`, `--seed`, `--no-augmentation` and `--no-pretrained`. Device options are `auto`, `cpu` and `cuda`.

Keep Colab's existing CUDA-compatible torch/torchvision installation (PyTorch >= 2.4); install only missing supporting packages. Do not run the local CPU-wheel installation cell on Colab. The online notebook, Drive access and hosted training were not tested during this refactor.

## Checkpoints and evaluation

The default outputs are `results/checkpoints/best_resnet18.pth` and `results/training_history.csv`. Best validation accuracy selects the checkpoint; accuracy is stored as a fraction from 0 to 1. History columns remain `epoch,train_loss,val_accuracy`. The existing `results/experiment_log.csv` is preserved but no longer written by training.

James can reload without downloading ImageNet weights:

```python
import torch
from src.model import load_checkpoint
from src.dataset import CountryDataset, create_loader

model, checkpoint = load_checkpoint("results/checkpoints/best_resnet18.pth", device="cpu")
test = CountryDataset("data/splits/test.csv", root_dir="data/processed",
                      country_to_idx=checkpoint["country_to_idx"])
images, labels = next(iter(create_loader(test)))
with torch.inference_mode():
    indices = model(images).argmax(dim=1).tolist()
predicted_countries = [checkpoint["countries"][index] for index in indices]
```

The helper returns an eval-mode model and the checkpoint. Retained keys are `architecture`, `model_state_dict`, `countries`, `country_to_idx`, `num_classes`, `epoch` and `best_val_accuracy`. Older checkpoints with extra metadata also load. New checkpoints omit optimizer/scaler state, configuration and normalization metadata; there is no resume feature. On CUDA evaluation, load onto CUDA and move input tensors there.

## Team workflow

Work on a branch and open a pull request for shared-file changes; get a teammate's review before merging. Pull regularly and resolve conflicts without discarding edits. Do not commit datasets or model weights; use the [shared Drive](https://drive.google.com/drive/folders/1ft7XsTZN0EtjHrBf-4OX56wlKJv9-i3v). Pin added dependencies and mention them in the PR. Codex does not commit or push automatically.
