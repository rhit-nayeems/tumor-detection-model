import glob, sys, numpy as np

d, expected = sys.argv[1], int(sys.argv[2])

files = sorted(glob.glob(f"{d}/chunk_*.npz"))

if len(files) != expected:

    raise SystemExit(f"Only {len(files)} of {expected} chunks done; rerun the missing ones first.")

parts = [np.load(f) for f in files]

out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}

np.savez(f"{d}/encoder_features.npz", **out)

print("features:", out["features"].shape,

      {s: int((out["split"] == s).sum()) for s in ("train", "val", "test")})
