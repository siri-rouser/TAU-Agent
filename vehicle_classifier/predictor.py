from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from .modeling import DinoVehicleAttributeModel


DEFAULT_EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


class VehicleClassifier:
    """Inference wrapper for checkpoints produced by train_modified.py."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        dinov2_repo_path: str | Path = "dinov2",
        ):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        config = ckpt["config"]
        self.colour_classes = list(ckpt["colour_classes"])
        self.type_classes = list(ckpt["type_classes"])
        self.car_subtype_classes = list(ckpt["car_subtype_classes"])

        self.model = DinoVehicleAttributeModel(
            backbone_name=config["backbone_name"],
            n_colours=len(self.colour_classes),
            n_types=len(self.type_classes),
            n_subtypes=len(self.car_subtype_classes),
            dinov2_repo_path=dinov2_repo_path,
        )
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.transform = DEFAULT_EVAL_TRANSFORM

    def _prepare_image(self, image: str | Path | Image.Image) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            if not image_path.is_file():
                raise IsADirectoryError(f"Image path must be a file: {image_path}")
            pil_img = Image.open(image_path).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError("image must be a path or PIL.Image.Image")

        tensor = self.transform(pil_img).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def results(self, image: str | Path | Image.Image) -> dict[str, Any]:
        """Return top-1 label, index, and confidence for each head."""
        batch = self._prepare_image(image)
        outputs = self.model(batch)

        colour_probs = torch.softmax(outputs["colour"], dim=1)[0]
        type_probs = torch.softmax(outputs["type"], dim=1)[0]
        subtype_probs = torch.softmax(outputs["car_subtype"], dim=1)[0]

        colour_idx = int(torch.argmax(colour_probs).item())
        type_idx = int(torch.argmax(type_probs).item())
        subtype_idx = int(torch.argmax(subtype_probs).item())

        return {
            "colour": {
                "index": colour_idx,
                "label": self.colour_classes[colour_idx],
                "confidence": float(colour_probs[colour_idx].item()),
            },
            "type": {
                "index": type_idx,
                "label": self.type_classes[type_idx],
                "confidence": float(type_probs[type_idx].item()),
            },
            "car_subtype": {
                "index": subtype_idx,
                "label": self.car_subtype_classes[subtype_idx],
                "confidence": float(subtype_probs[subtype_idx].item()),
            },
        }


def vehicle_classifier(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    dinov2_repo_path: str | Path = "dinov2",
) -> VehicleClassifier:
    """Factory function requested API: vehicle_classifier('xxx.pt')."""
    return VehicleClassifier(
        checkpoint_path=checkpoint_path,
        device=device,
        dinov2_repo_path=dinov2_repo_path,
    )
