# Encoder features



`features/encoder_features.npz` holds one 640-dim vector per scan (mean + max pooled

deepest encoder features of the autoPET III ResEncL model, fold 0, CHECKPOINT_NAME).

Keys: features, case_ids, split (train/val/test), tracer (fdg/psma), lesion (1 = tumor).

The split matches `data/split_60_20_20.json` (patient-level, test set never trained on).



Load: `d = np.load("features/encoder_features.npz"); X, y = d["features"], d["lesion"]`



Reproducing: MIC-DKFZ/autopet-3-submission with two patches for newer PyTorch:

- `nnunetv2/training/lr_scheduler/polylr.py`: remove the trailing `, False` in `super().__init__`

- set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (in the sbatch files)
