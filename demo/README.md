---
title: PET CT Classifier Demo
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
python_version: "3.12"
---

# Local demo

## GitHub Pages version

`docs/` contains the browser version of this demo. It uses the same v2 weights,
training mean/std, and 0.55 threshold. JavaScript runs the 640 -> 16 -> 1 head
when Predict is clicked; predictions are not stored in a lookup table.

To preview from the project root:

```powershell
.venv/Scripts/python.exe -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open http://127.0.0.1:8000. The page must be served over HTTP, rather than
opened as a local file, so the browser can load the JSON assets.

For GitHub Pages, publish branch `main`, folder `/docs`, from the personal
repository's Settings > Pages. The source and JSON assets are already in `docs/`;
there is no Python runtime or build step on the host. Keep the team repository's
remote separate from the personal repository.

`demo/export_pages.py` regenerates the browser assets from the local v2 checkpoint
and original feature archive. It needs PyTorch and NumPy, but the hosted page does
not. `docs/model.json` includes weights and normalization; `docs/scans.json`
contains only aliased validation features and labels. No patient IDs, dates,
training samples, or test samples are exported. These assets are public when
the site is published.

Checked all 315 browser predictions against PyTorch: every binary prediction
matched, and the maximum probability difference was below 0.000001. Small
floating-point differences are expected between the two runtimes.

## Gradio version

Choose a validation case to run the saved v2 classification head. The app shows
the tumor score, thresholded prediction, known label, and whether they match.
Features were extracted earlier. Raw scans, image previews, uploads, and heatmaps
are not included. It runs on CPU and does not retrain anything.

From the project root in PowerShell:

```powershell
uv pip install --python .venv/Scripts/python.exe -r demo/requirements.txt
.venv/Scripts/python.exe demo/prepare_demo.py
.venv/Scripts/python.exe demo/app.py
```

Open http://127.0.0.1:7860. Press Ctrl+C in the terminal to stop the app.
Preparation requires the existing local `checkpoints/improved_v2.pt` and
`features/encoder_features.npz`. It checks the feature archive hash before
copying the checkpoint and exporting only validation cases with display aliases.

## Hosting later

Nothing has been uploaded. The branch and assets are local.

For a Hugging Face Gradio Space, copy the contents of this folder into the Space
root: `app.py`, `requirements.txt`, `README.md`, and the prepared `assets/` folder.
Create a private Space first. Use Python 3.12 and CPU hardware. The README header
configures the SDK; the runtime installs requirements and runs `app.py`.
Do not upload the full project, raw scan files, or the original feature archive.

The local assets are ignored by Git. Include them explicitly when preparing the
Space upload; the classifier checkpoint is not otherwise in the repository.
No access token is needed to run this locally.

The SDK setup follows the [Gradio Spaces documentation](https://huggingface.co/docs/hub/spaces-sdks-gradio).

## Current result

V2 validation accuracy is 72.06%, balanced accuracy 71.51%, and its tumor
threshold is 0.55. The head has 640 inputs, 16 ReLU hidden neurons, and one logit.
Training normalization statistics are loaded from the checkpoint at inference.

All 315 validation examples are included, so the demo contains correct and
incorrect predictions. Validation was used to choose the model and threshold;
these examples are not an independent accuracy assessment. The original encoder
checkpoint remains unconfirmed. Future feature extraction fixes have not been
applied to this archive. This is a course demo, not a medical tool.

## Local checks

The page returned HTTP 200. All 315 validation predictions matched the saved v2
checkpoint. The web API was checked with correct and incorrect examples of both
classes. The export was checked against the validation split and contains no
original case IDs or test examples. Browser visual review was unavailable because
local browser control is disabled in Work settings.
