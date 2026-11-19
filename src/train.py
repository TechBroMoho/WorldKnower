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
    from .geo_utils import distance_matrix, geoguessr_score
    from .model import ARCHITECTURE, ARCHITECTURES, HEAD_NAMES, build_model
else:
    from dataset import create_dataloaders
    from geo_utils import distance_matrix, geoguessr_score
    from model import ARCHITECTURE, ARCHITECTURES, HEAD_NAMES, build_model


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="data/splits/train.csv")
    parser.add_argument("--val-csv", default="data/splits/val.csv")
    parser.add_argument("--dataset-root", default=".")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--history-path", default="results/training_history.csv")
    parser.add_argument("--loss", choices=["ce", "distance"], default="ce",
                        help="ce: cross-entropy; distance: cross-entropy + expected-km penalty")
    parser.add_argument("--distance-weight", type=float, default=1.0,
                        help="weight on expected_km / 1000 when --loss distance")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", "--learning-rate", type=float, default=0.0001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augmentation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--architecture", choices=sorted(ARCHITECTURES), default=ARCHITECTURE)
    parser.add_argument("--select-by", choices=["accuracy", "score"], default="accuracy",
                        help="validation metric that picks the saved checkpoint (score: mean GeoGuessr score)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.epochs < 1 or args.lr <= 0:
        raise ValueError("epochs and learning rate must be positive")
    if args.distance_weight < 0:
        raise ValueError("distance weight must be non-negative")
    if args.select_by == "score" and args.loss == "ce":
        print("Note: selecting by GeoGuessr score while training with plain cross-entropy.", flush=True)
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
    model = build_model(len(countries), args.pretrained, args.freeze_backbone, args.architecture).to(device)
    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad], lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    use_distance = args.loss == "distance"
    try:
        dist_matrix = torch.from_numpy(distance_matrix(countries)).to(device)  # (N, N) km, label order
    except KeyError:  # synthetic labels (smoke test) have no coordinates
        if use_distance:
            raise
        dist_matrix = None
    if args.select_by == "score" and dist_matrix is None:
        raise ValueError("--select-by score needs country coordinates for every label")

    def compute_loss(logits, labels):
        loss = criterion(logits, labels)
        if use_distance:
            probs = logits.float().softmax(dim=1)
            expected_km = (probs * dist_matrix[labels]).sum(dim=1).mean()
            loss = loss + args.distance_weight * expected_km / 1000
        return loss

    print(f"Loss: {args.loss}" + (f" (weight {args.distance_weight})" if use_distance else ""), flush=True)
    amp = args.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    checkpoint_path = Path(args.checkpoint_dir) / f"best_{args.architecture}.pth"
    history_path = Path(args.history_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = best_score = best_metric = -1.0

    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.writer(history_file)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "val_mean_km", "val_geoguessr_score"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            if args.freeze_backbone:
                for name, module in model.named_children():
                    if name != HEAD_NAMES[args.architecture]:
                        module.eval()
            total_loss = 0.0
            for images, labels in train_loader:
                images = images.to(device, non_blocking=device == "cuda")
                labels = labels.to(device, non_blocking=device == "cuda")
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device, enabled=amp):
                    loss = compute_loss(model(images), labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * labels.size(0)
            train_loss = total_loss / len(train_loader.dataset)

            model.eval()
            correct = 0
            val_loss_sum = km_sum = score_sum = 0.0
            with torch.inference_mode():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=device == "cuda")
                    labels = labels.to(device, non_blocking=device == "cuda")
                    logits = model(images)
                    val_loss_sum += compute_loss(logits, labels).item() * labels.size(0)
                    predictions = logits.argmax(dim=1)
                    correct += (predictions == labels).sum().item()
                    if dist_matrix is not None:
                        km = dist_matrix[labels, predictions]
                        km_sum += km.sum().item()
                        score_sum += geoguessr_score(km.cpu().numpy()).sum()
            accuracy = correct / len(val_loader.dataset)
            val_loss = val_loss_sum / len(val_loader.dataset)
            val_km = km_sum / len(val_loader.dataset) if dist_matrix is not None else float("nan")
            val_score = float(score_sum) / len(val_loader.dataset) if dist_matrix is not None else float("nan")
            print(f"Epoch {epoch}/{args.epochs} | Training loss: {train_loss:.4f} | Validation loss: {val_loss:.4f}"
                  f" | Validation accuracy: {accuracy:.2%}"
                  f" | Mean distance: {val_km:.0f} km | GeoGuessr score: {val_score:.0f}", flush=True)
            writer.writerow([epoch, train_loss, val_loss, accuracy, val_km, val_score])
            history_file.flush()
            metric = val_score if args.select_by == "score" else accuracy
            if metric > best_metric:
                best_metric, best_accuracy, best_score = metric, accuracy, val_score
                torch.save(dict(architecture=args.architecture, model_state_dict=model.state_dict(),
                                countries=countries, country_to_idx=mapping, num_classes=len(countries),
                                epoch=epoch, best_val_accuracy=best_accuracy, best_val_geoguessr_score=best_score,
                                selected_by=args.select_by, loss=args.loss,
                                freeze_backbone=args.freeze_backbone), checkpoint_path)
                print(f"Saved best checkpoint (by {args.select_by}): {checkpoint_path}", flush=True)
    return dict(checkpoint_path=str(checkpoint_path), history_path=str(history_path),
                best_val_accuracy=best_accuracy, best_val_geoguessr_score=best_score)


if __name__ == "__main__":
    main()
