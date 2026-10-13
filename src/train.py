"""Train the country classifier. Run as python -m src.train or python src/train.py."""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import random
import uuid

import numpy as np
import torch
from torch import nn

if __package__:
    from .dataset import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, create_dataloaders
    from .model import ARCHITECTURE, build_model
else:
    from dataset import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, create_dataloaders
    from model import ARCHITECTURE, build_model


def select_device(mode="auto"):
    if mode not in {"auto", "cuda", "cpu"}:
        raise ValueError("device must be auto, cuda, or cpu")
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Select a hosted Colab GPU kernel; use --device cpu for local tests.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed, device, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    # Fast fixed-size convolutions by default; opt into stricter repeatability.
    torch.backends.cudnn.benchmark = device.type == "cuda" and not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def train_epoch(model, loader, optimizer, criterion, device, scaler, amp):
    model.train()
    total_loss = torch.zeros((), device=device)
    count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.detach() * labels.size(0)
        count += labels.size(0)
    return total_loss.item() / count


@torch.inference_mode()
def validation_accuracy(model, loader, device, amp=False):
    model.eval()
    correct = torch.zeros((), dtype=torch.long, device=device)
    count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum()
        count += labels.size(0)
    return correct.item() / count


def append_experiment(path, args, best_accuracy):
    """Append a completed run using the existing schema; F1 belongs to evaluation."""
    fields = ["run_id", "date", "author", "model", "epochs", "batch_size", "lr",
              "val_accuracy", "val_f1", "notes"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="", encoding="utf-8") as file:
            if next(csv.reader(file), []) != fields:
                raise ValueError(f"Unexpected experiment log schema: {path}")
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(dict(run_id=uuid.uuid4().hex, date=datetime.now(timezone.utc).isoformat(),
                             author=args.author, model=ARCHITECTURE, epochs=args.epochs,
                             batch_size=args.batch_size, lr=args.lr, val_accuracy=best_accuracy,
                             val_f1="", notes=args.notes))


def run_training(args):
    if args.epochs < 1 or args.lr <= 0:
        raise ValueError("epochs and learning rate must be positive")
    device = select_device(args.device)
    seed_everything(args.seed, device, args.deterministic)
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    amp = bool(args.amp and device.type == "cuda"
               and torch.amp.autocast_mode.is_autocast_available("cuda"))
    print(f"Mixed precision: {amp}", flush=True)
    train_loader, val_loader = create_dataloaders(
        args.train_csv, args.val_csv, args.dataset_root, args.batch_size,
        args.num_workers, args.augmentation, device, args.seed)
    mapping = train_loader.dataset.country_to_idx
    countries = train_loader.dataset.countries
    print(f"Number of countries: {len(countries)}", flush=True)
    model = build_model(len(countries), args.pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    checkpoint_dir = Path(args.checkpoint_dir)
    history_path = Path(args.history_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_resnet18.pth"
    best_accuracy = -1.0  # Save the first epoch even when accuracy is zero.
    history = []
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=["epoch", "train_loss", "val_accuracy"])
        writer.writeheader()
        history_file.flush()
        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler, amp)
            accuracy = validation_accuracy(model, val_loader, device, amp)
            print(f"\nEpoch {epoch}/{args.epochs}\nTraining Loss: {loss:.4f}\nValidation Accuracy: {accuracy:.2%}", flush=True)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                checkpoint = dict(
                    architecture=ARCHITECTURE, model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(), scaler_state_dict=scaler.state_dict(),
                    country_to_idx=mapping, countries=countries, num_classes=len(countries),
                    epoch=epoch, best_val_accuracy=best_accuracy,
                    config=vars(args).copy(), image_size=IMAGE_SIZE,
                    normalization=dict(mean=IMAGENET_MEAN, std=IMAGENET_STD))
                # A failed write leaves the prior best checkpoint intact.
                temporary = checkpoint_path.with_suffix(".pth.tmp")
                torch.save(checkpoint, temporary)
                temporary.replace(checkpoint_path)
                print(f"Saved best checkpoint: {checkpoint_path}", flush=True)
            row = dict(epoch=epoch, train_loss=loss, val_accuracy=accuracy)
            history.append(row)
            writer.writerow(row)
            history_file.flush()
    if args.experiment_log:
        append_experiment(args.experiment_log, args, best_accuracy)
    return dict(checkpoint_path=str(checkpoint_path), history_path=str(history_path),
                best_val_accuracy=best_accuracy, history=history, country_to_idx=mapping)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="data/splits/train.csv")
    parser.add_argument("--val-csv", default="data/splits/val.csv")
    parser.add_argument("--dataset-root", default=".")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", "--learning-rate", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augmentation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--history-path", default="results/training_history.csv")
    parser.add_argument("--experiment-log", default=None, help="Optional append-only experiment CSV")
    parser.add_argument("--author", default="Swapnil")
    parser.add_argument("--notes", default="")
    return parser


def main(argv=None):
    return run_training(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
