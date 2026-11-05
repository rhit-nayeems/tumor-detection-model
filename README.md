# tumor-detection-model

Binary tumor classification from 3D PET/CT encoder features.

This is a personal fork of the [team project](https://github.com/rhit-steelevw/tumor-detection-model).
The team commit history and authors are retained. The browser demo uses the
saved v2 classification head and precomputed AutoPET validation features.

## Demo

`docs/` is the GitHub Pages site. To run it locally from the project root:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open http://127.0.0.1:8000, choose a scan, and click Predict. The classifier runs
in the browser and shows tumor/no-tumor probabilities, its prediction, and the
known label. It does not accept new scans or run the full encoder.

The head uses 640 features, 16 ReLU hidden neurons, and one output, with the
saved training normalization and a tumor threshold of 0.55. Its validation
accuracy is 72.06%, balanced accuracy 71.51%. Validation was used for model
selection; these are development results. PSMA-negative cases remain a weakness.

## Project files

- `classifier.ipynb`: baseline, classification heads, training, and evaluation.
- `encoder/`: server scripts for preprocessing, training, and feature extraction.
- `features/encoder_features.npz`: the original saved encoder features.
- `demo/`: local Gradio app, browser exporter, and setup instructions.
- `docs/`: static site and exported model/validation assets.

Model checkpoint files in `checkpoints/` are local and ignored by Git. The site
has its own exported weights; it does not need those Python checkpoint files.
See [demo setup](demo/README.md) and [encoder instructions](encoder/README.md).

Raw PET/CT scans and the encoder checkpoint are not included. The exact encoder
checkpoint behind the original feature archive still needs confirmation.
