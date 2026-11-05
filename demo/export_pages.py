"""Export the saved head and validation features for browser inference."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


root = Path(__file__).resolve().parent.parent
source = root / "features/encoder_features.npz"
checkpoint = torch.load(root / "checkpoints/improved_v2.pt", map_location="cpu", weights_only=True)
if hashlib.sha256(source.read_bytes()).hexdigest() != checkpoint["features_sha256"]:
    raise ValueError("Features differ from the archive used to train version 2")

weights = checkpoint["model_state_dict"]
model = {
    "inputSize": checkpoint["in_features"],
    "hiddenSize": checkpoint["config"]["hidden_size"],
    "threshold": checkpoint["threshold"],
    "mean": checkpoint["feature_mean"].flatten().tolist(),
    "std": checkpoint["feature_std"].flatten().tolist(),
    "hiddenWeight": weights["0.weight"].tolist(),
    "hiddenBias": weights["0.bias"].tolist(),
    "outputWeight": weights["2.weight"].flatten().tolist(),
    "outputBias": weights["2.bias"].item(),
}
with np.load(source, allow_pickle=False) as data:
    mask = data["split"] == "val"
    scans = [{"name": f"Scan {i + 1:03d}", "features": features.tolist(), "actual": int(label)}
             for i, (features, label) in enumerate(zip(data["features"][mask], data["lesion"][mask]))]

destination = root / "docs"
destination.mkdir(exist_ok=True)
for name, content in (("model.json", model), ("scans.json", scans)):
    (destination / name).write_text(json.dumps(content, separators=(",", ":"), allow_nan=False), encoding="utf-8")
print(f"Exported model and {len(scans)} validation scans to {destination}")
