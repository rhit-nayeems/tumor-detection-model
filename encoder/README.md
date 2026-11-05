# Encoder features



`features/encoder_features.npz` holds one 640-dim vector per scan (mean + max pooled

deepest encoder features of the autoPET III ResEncL model, fold 0, CHECKPOINT_NAME).

Keys: features, case_ids, split (train/val/test), tracer (fdg/psma), lesion (1 = tumor).

The split matches `data/split_60_20_20.json` (patient-level, test set never trained on).



Load: `d = np.load("features/encoder_features.npz"); X, y = d["features"], d["lesion"]`



Reproducing: MIC-DKFZ/autopet-3-submission with two patches for newer PyTorch:

- `nnunetv2/training/lr_scheduler/polylr.py`: remove the trailing `, False` in `super().__init__`

- set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (in the sbatch files)


## Running extraction

The existing feature archive does not record its encoder checkpoint. The checks
below apply to new runs.

After training finishes writing the checkpoint, copy it to a filename training
will not overwrite. Set MODEL to the model directory on the server.

```bash
export ENCODER_CHECKPOINT=checkpoint_for_features_v1.pth
cp -n "$MODEL/fold_0/checkpoint_final.pth" "$MODEL/fold_0/$ENCODER_CHECKPOINT"
export ENCODER_CHECKPOINT_SHA256=$(sha256sum "$MODEL/fold_0/$ENCODER_CHECKPOINT" | cut -d ' ' -f 1)
export FEATURE_OUT="$AP/features_v1"
sbatch --export=ALL "$AP/jobs/encode.sbatch"
```

Copy `feature_checks.py`, `extract_features.py`, and `merge_features.py` to `$AP`
and update `jobs/encode.sbatch`. Use a fresh output directory. Every task checks
the same checkpoint hash and records hashes of the plans, split, and code.
Cached chunks are reused only if their metadata and cases match.

Once all jobs finish:

```bash
python "$AP/merge_features.py" "$FEATURE_OUT" 20 --split "$AP/split_60_20_20.json"
```

The merger checks chunk membership, hashes, dimensions, labels, and finite
features before writing the output. Old chunks without metadata need to be
re-extracted. Keep the old archive and classifier checkpoints for comparison.

The updated extraction still needs a run on the server with the real data.
