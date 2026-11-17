"""Model factory for the supported architectures and checkpoint reconstruction interface."""

import torch
from torch import nn
from torchvision.models import (EfficientNet_B0_Weights, ResNet18_Weights, ResNet50_Weights,
                                efficientnet_b0, resnet18, resnet50)

ARCHITECTURE = "resnet18"
ARCHITECTURES = {
    "resnet18": (resnet18, ResNet18_Weights, "fc"),
    "resnet50": (resnet50, ResNet50_Weights, "fc"),
    "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights, "classifier"),
}
HEAD_NAMES = {name: spec[2] for name, spec in ARCHITECTURES.items()}


def build_model(num_classes, pretrained=True, freeze_backbone=False, architecture=ARCHITECTURE):
    """Full fine-tuning by default; disable weights for offline tests/reloading."""
    if num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {architecture}. Choose from {sorted(ARCHITECTURES)}")
    factory, weights, head_name = ARCHITECTURES[architecture]
    model = factory(weights=weights.DEFAULT if pretrained else None)
    #skips over the tensor during backpropogation so it won't compute the gradient
    #weight stays at pretrained value which speeds up training steps
    #ensures that only the final layer gets trained
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    if head_name == "fc":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def load_checkpoint(checkpoint_path, device="cpu"):
    """Return (model in eval mode, checkpoint metadata), without downloading weights."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("architecture")
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
    model = build_model(checkpoint["num_classes"], pretrained=False, architecture=architecture)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint
