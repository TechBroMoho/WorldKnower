"""Fine-tune ResNet18: python -m src.train --help."""

import argparse
import csv
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

if __package__:
    from .dataset import create_dataloaders
    from .model import ARCHITECTURE, build_model
else:
    from dataset import create_dataloaders
    from model import ARCHITECTURE, build_model


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="data/splits/train.csv")
    parser.add_argument("--val-csv", default="data/splits/val.csv")
    parser.add_argument("--dataset-root", default=".")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--history-path", default="results/training_history.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", "--learning-rate", type=float, default=0.0001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augmentation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.epochs < 1 or args.lr <= 0:
        raise ValueError("epochs and learning rate must be positive")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Select a hosted Colab GPU kernel.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    train_loader, val_loader = create_dataloaders(
        args.train_csv, args.val_csv, args.dataset_root, args.batch_size,
        args.num_workers, args.augmentation, device, args.seed)
    countries = train_loader.dataset.countries
    mapping = train_loader.dataset.country_to_idx
    print(f"Number of countries: {len(countries)}", flush=True)
    model = build_model(len(countries), args.pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    amp = args.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    checkpoint_path = Path(args.checkpoint_dir) / "best_resnet18.pth"
    history_path = Path(args.history_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0

    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.writer(history_file)
        writer.writerow(["epoch", "train_loss", "val_accuracy"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            for images, labels in train_loader:
                images = images.to(device, non_blocking=device == "cuda")
                labels = labels.to(device, non_blocking=device == "cuda")
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device, enabled=amp):
                    loss = criterion(model(images), labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * labels.size(0)
            train_loss = total_loss / len(train_loader.dataset)

            model.eval()
            correct = 0
            with torch.inference_mode():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=device == "cuda")
                    labels = labels.to(device, non_blocking=device == "cuda")
                    correct += (model(images).argmax(dim=1) == labels).sum().item()
            accuracy = correct / len(val_loader.dataset)
            print(f"Epoch {epoch}/{args.epochs} | Training loss: {train_loss:.4f} | Validation accuracy: {accuracy:.2%}", flush=True)
            writer.writerow([epoch, train_loss, accuracy])
            history_file.flush()
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(dict(architecture=ARCHITECTURE, model_state_dict=model.state_dict(),
                                countries=countries, country_to_idx=mapping, num_classes=len(countries),
                                epoch=epoch, best_val_accuracy=best_accuracy), checkpoint_path)
                print(f"Saved best checkpoint: {checkpoint_path}", flush=True)
    return dict(checkpoint_path=str(checkpoint_path), history_path=str(history_path),
                best_val_accuracy=best_accuracy)


if __name__ == "__main__":
    main()
