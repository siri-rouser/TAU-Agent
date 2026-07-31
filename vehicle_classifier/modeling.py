from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import cast

import torch
import torch.nn as nn


class DinoVehicleAttributeModel(nn.Module):
    """DINOv2 backbone plus three supervised heads."""

    FEATURE_DIMS = {
        "dinov2_vits14": 384,
        "dinov2_vits14_reg": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitb14_reg": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitl14_reg": 1024,
        "dinov2_vitg14": 1536,
        "dinov2_vitg14_reg": 1536,
    }

    def __init__(
        self,
        backbone_name: str,
        n_colours: int,
        n_types: int,
        n_subtypes: int,
        hidden_dim: int = 512,
        dinov2_repo_path: str | Path = "dinov2",
    ):
        super().__init__()
        if backbone_name not in self.FEATURE_DIMS:
            raise ValueError(
                f"Unknown backbone {backbone_name!r}. Known options: {sorted(self.FEATURE_DIMS)}"
            )

        repo_path = Path(dinov2_repo_path)
        if not repo_path.exists():
            raise FileNotFoundError(f"DINOv2 repo path not found: {repo_path}")

        load_kwargs: dict[str, Any] = {
            "source": "local",
            # Backbone parameters are restored from checkpoint via load_state_dict.
            # Keep hub initialization uninitialized to avoid external weight downloads.
            "pretrained": False,
        }

        self.backbone = cast(nn.Module, torch.hub.load(str(repo_path), backbone_name, **load_kwargs))
        feat_dim = self.FEATURE_DIMS[backbone_name]

        self.norm = nn.LayerNorm(feat_dim)
        self.colour_head = self._make_head(feat_dim, hidden_dim, n_colours)
        self.type_head = self._make_head(feat_dim, hidden_dim, n_types)
        self.car_subtype_head = self._make_head(feat_dim, hidden_dim, n_subtypes)

    @staticmethod
    def _make_head(feat_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        features = self.norm(features)
        return {
            "colour": self.colour_head(features),
            "type": self.type_head(features),
            "car_subtype": self.car_subtype_head(features),
        }
