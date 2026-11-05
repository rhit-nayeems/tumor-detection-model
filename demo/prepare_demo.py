"""Copy v2 and validation-only features into the local demo folder."""

import hashlib
import shutil
from pathlib import Path

import numpy as np
import torch


root = Path(__file__).resolve().parent.parent
source = root / "features" / "encoder_features.npz"
checkpoint_path = root / "checkpoints" / "improved_v2.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
if hashlib.sha256(source.read_bytes()).hexdigest() != checkpoint["features_sha256"]:
    raise ValueError("Feature archive differs from the one used to train version 2")

with np.load(source, allow_pickle=False) as data:
    validation = data["split"] == "val"
    features = data["features"][validation]
    labels = data["lesion"][validation]
    tracers = data["tracer"][validation]

assets = Path(__file__).resolve().parent / "assets"
assets.mkdir(exist_ok=True)
shutil.copy2(checkpoint_path, assets / "classifier.pt")
np.savez_compressed(assets / "validation_cases.npz", features=features,
                    lesion=labels, tracer=tracers,
                    case_names=np.array([f"Scan {i + 1:03d}" for i in range(len(labels))]))
print(f"Prepared {len(labels)} validation cases in {assets}")
print("Patient IDs, dates, training cases, and test cases are not included.")
