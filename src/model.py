"""The single ResNet18 factory and checkpoint reconstruction interface."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

ARCHITECTURE = "resnet18"


def build_model(num_classes, pretrained=True):
    """Full fine-tuning by default; disable weights for offline tests/reloading."""
    if num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(checkpoint_path, device="cpu"):
    """Return (model in eval mode, checkpoint metadata), without downloading weights."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError(f"Unsupported checkpoint architecture: {checkpoint.get('architecture')}")
    model = build_model(checkpoint["num_classes"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint
