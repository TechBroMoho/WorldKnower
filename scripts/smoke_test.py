"""Test the real pipeline on CPU using small temporary synthetic image splits."""

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.dataset import CountryDataset, build_transforms, create_dataloaders, create_loader
from src.model import build_model, load_checkpoint
from src.train import main


def run():
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory(prefix="worldknower_smoke_") as directory:
        root = Path(directory)
        train_rows, val_rows = [], []
        countries = ["CountryA", "CountryB", "CountryC"]
        for country in countries:
            for index in range(3):
                relative = Path("images") / country / f"{index}.png"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)).save(path)
                row = dict(path=relative.as_posix(), country=country)
                (train_rows if index < 2 else val_rows).append(row)
        pd.DataFrame(train_rows).to_csv(root / "train.csv", index=False)
        pd.DataFrame(val_rows[::-1]).to_csv(root / "val.csv", index=False)
        train, val = create_dataloaders(root / "train.csv", root / "val.csv", root,
                                        batch_size=3, augmentation=False, device="cpu")
        mapping = {country: index for index, country in enumerate(countries)}
        assert train.dataset.country_to_idx == val.dataset.country_to_idx == mapping
        assert val.dataset[0][1] == mapping["CountryC"]
        assert torch.equal(val.dataset[0][0], val.dataset[0][0])
        images, labels = next(iter(train))
        assert images.shape == (3, 3, 224, 224) and labels.dtype == torch.int64
        assert not train.pin_memory
        with Image.open(train.dataset.image_paths[0]) as image:
            assert build_transforms(True)(image).shape == (3, 224, 224)
        Image.fromarray(rng.integers(0, 256, (32, 32), dtype=np.uint8)).save(root / "gray.png")
        pd.DataFrame([dict(path=str(root / "gray.png"), country="CountryA")]).to_csv(root / "gray.csv", index=False)
        gray = CountryDataset(root / "gray.csv", country_to_idx=mapping)
        assert gray[0][0].shape == (3, 224, 224)  # Also checks absolute paths and RGB conversion.
        worker_loader = create_loader(val.dataset, batch_size=3, num_workers=1, device="cpu")
        assert next(iter(worker_loader))[0].shape == (3, 3, 224, 224)

        torch.manual_seed(42)
        model = build_model(3, pretrained=False)
        initial_weights = model.conv1.weight.detach().clone()
        logits = model(images)
        assert logits.shape == (3, 3)
        torch.nn.functional.cross_entropy(logits, labels).backward()
        assert model.conv1.weight.grad is not None and torch.isfinite(model.conv1.weight.grad).all()
        del model
        result = main([
            "--train-csv", str(root / "train.csv"), "--val-csv", str(root / "val.csv"),
            "--dataset-root", str(root), "--epochs", "1", "--batch-size", "3",
            "--device", "cpu", "--no-pretrained", "--no-augmentation",
            "--checkpoint-dir", str(root / "checkpoints"), "--history-path", str(root / "history.csv"),
        ])
        loaded, checkpoint = load_checkpoint(result["checkpoint_path"], device="cpu")
        assert checkpoint["country_to_idx"] == mapping and checkpoint["countries"] == countries
        assert checkpoint["num_classes"] == 3 and checkpoint["epoch"] == 1
        assert checkpoint["architecture"] == "resnet18" and 0 <= checkpoint["best_val_accuracy"] <= 1
        assert not torch.equal(initial_weights, loaded.conv1.weight)  # Full backbone trained.
        with torch.inference_mode():
            assert loaded(images).shape == (3, 3)
        history = pd.read_csv(result["history_path"])
        assert list(history.columns) == ["epoch", "train_loss", "val_accuracy"] and len(history) == 1
        assert np.isfinite(history.train_loss.iloc[0])
        assert history.val_accuracy.iloc[0] == result["best_val_accuracy"]
        print("PASS: RGB loading, transforms, mappings, batches, Windows worker, forward/backward, CPU epoch, validation, checkpoint reload.")
    print("Temporary test images and checkpoints removed.")


if __name__ == "__main__":
    run()
