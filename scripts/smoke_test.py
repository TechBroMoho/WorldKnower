"""Small CPU-only integration test; every generated artifact lives in a temp directory."""

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.dataset import CountryDataset, build_transforms, create_dataloaders, create_loader, stage_selected_data
from src.model import build_model, load_checkpoint
from src.train import main, select_device


def expect_error(exception, callback, text):
    try:
        callback()
    except exception as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError(f"Expected {exception.__name__}: {text}")


def run():
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory(prefix="worldknower_smoke_") as directory:
        root = Path(directory)
        countries = ["CountryA", "CountryB", "CountryC"]
        rows = []
        for country in countries:
            for index in range(3):
                relative = Path("images") / country / f"image{index:03}.png"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                pixels = rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(path)
                rows.append(dict(path=relative.as_posix(), country=country))
        train_rows = [row for i, row in enumerate(rows) if i % 3 != 2]
        val_rows = [row for i, row in enumerate(rows) if i % 3 == 2][::-1]
        pd.DataFrame(train_rows).to_csv(root / "train.csv", index=False)
        pd.DataFrame(val_rows).to_csv(root / "val.csv", index=False)
        train, val = create_dataloaders(root / "train.csv", root / "val.csv", root,
                                        batch_size=3, augmentation=False, device="cpu")
        mapping = {country: i for i, country in enumerate(countries)}
        assert train.dataset.country_to_idx == val.dataset.country_to_idx == mapping
        image, label = train.dataset[0]
        assert image.shape == (3, 224, 224) and image.dtype == torch.float32 and label == 0
        assert torch.equal(val.dataset[0][0], val.dataset[0][0])
        images, labels = next(iter(train))
        assert images.shape == (3, 3, 224, 224) and labels.dtype == torch.int64
        assert not train.pin_memory

        # Reverse/subset validation must retain training indices.
        pd.DataFrame([val_rows[0]]).to_csv(root / "subset.csv", index=False)
        subset = CountryDataset(root / "subset.csv", root, mapping)
        assert subset[0][1] == mapping["CountryC"]
        # Exercise RGB conversion for non-RGB sources, absolute paths and CSV-parent root.
        gray = root / "gray.png"
        Image.fromarray(rng.integers(0, 256, (32, 32), dtype=np.uint8)).save(gray)
        rgba = root / "rgba.png"
        Image.fromarray(rng.integers(0, 256, (32, 32, 4), dtype=np.uint8)).save(rgba)
        pd.DataFrame([dict(path=str(gray), country="CountryA"),
                      dict(path="rgba.png", country="CountryB")]).to_csv(root / "modes.csv", index=False)
        modes = CountryDataset(root / "modes.csv", country_to_idx=mapping)
        assert modes[0][0].shape == modes[1][0].shape == (3, 224, 224)
        assert build_transforms(True, True)(Image.open(rgba).convert("RGB")).shape == (3, 224, 224)

        pd.DataFrame([dict(path="missing.png", country="CountryA")]).to_csv(root / "bad.csv", index=False)
        expect_error(FileNotFoundError, lambda: CountryDataset(root / "bad.csv", root, mapping), "missing image")
        pd.DataFrame([dict(path="gray.png", country="Unknown")]).to_csv(root / "bad.csv", index=False)
        expect_error(ValueError, lambda: CountryDataset(root / "bad.csv", root, mapping), "unknown countries")
        pd.DataFrame([dict(filename="gray.png")]).to_csv(root / "bad.csv", index=False)
        expect_error(ValueError, lambda: CountryDataset(root / "bad.csv"), "must contain columns")
        corrupt = root / "corrupt.png"
        corrupt.write_text("not an image")
        pd.DataFrame([dict(path="corrupt.png", country="CountryA")]).to_csv(root / "bad.csv", index=False)
        expect_error(ValueError, lambda: CountryDataset(root / "bad.csv", root, mapping)[0], "Cannot decode image")
        assert select_device("cpu").type == "cpu"
        if not torch.cuda.is_available():
            expect_error(RuntimeError, lambda: select_device("cuda"), "CUDA requested but unavailable")

        # A spawned worker verifies Windows pickling and seeding, without training on GPU.
        worker_loader = create_loader(val.dataset, batch_size=3, num_workers=1, device="cpu")
        assert next(iter(worker_loader))[0].shape == (3, 3, 224, 224)
        staged_root, staged_train, staged_val = stage_selected_data(
            root / "train.csv", root / "val.csv", root, root / "staged")
        staged = CountryDataset(staged_val, staged_root, mapping)
        assert len(staged) == 3 and torch.equal(staged[0][0], val.dataset[0][0])
        assert len(list((staged_root / "images").iterdir())) == 9

        torch.manual_seed(42)
        model = build_model(3, pretrained=False)
        initial_backbone = model.conv1.weight.detach().clone()
        logits = model(images)
        assert logits.shape == (3, 3)
        torch.nn.functional.cross_entropy(logits, labels).backward()
        assert model.conv1.weight.grad is not None and torch.isfinite(model.conv1.weight.grad).all()
        del model
        result = main([
            "--train-csv", str(staged_train), "--val-csv", str(staged_val),
            "--dataset-root", str(staged_root), "--epochs", "1", "--batch-size", "3",
            "--device", "cpu", "--no-pretrained", "--no-augmentation",
            "--checkpoint-dir", str(root / "checkpoints"), "--history-path", str(root / "history.csv"),
            "--experiment-log", str(root / "experiments.csv"), "--deterministic",
        ])
        assert 0 <= result["best_val_accuracy"] <= 1
        loaded, checkpoint = load_checkpoint(result["checkpoint_path"], device="cpu")
        assert checkpoint["country_to_idx"] == mapping and checkpoint["countries"] == countries
        assert checkpoint["num_classes"] == 3 and checkpoint["epoch"] == 1
        assert checkpoint["architecture"] == "resnet18" and checkpoint["config"]["device"] == "cpu"
        assert checkpoint["config"]["pretrained"] is False
        assert not torch.equal(initial_backbone, loaded.conv1.weight)
        with torch.inference_mode():
            assert loaded(images).shape == (3, 3)
        history = pd.read_csv(root / "history.csv")
        assert list(history.columns) == ["epoch", "train_loss", "val_accuracy"]
        assert len(history) == 1 and np.isfinite(history.train_loss.iloc[0])
        assert history.val_accuracy.iloc[0] == result["best_val_accuracy"]
        assert len(pd.read_csv(root / "experiments.csv")) == 1
        print("PASS: dataset, transforms, mappings, errors, Windows worker, staging, forward/backward, CPU epoch, history, checkpoint reload.")
        print("All synthetic artifacts were isolated in temporary storage and are now removed.")


if __name__ == "__main__":
    run()
