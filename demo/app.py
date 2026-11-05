import os
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import numpy as np
import torch
from torch import nn


assets = Path(__file__).resolve().parent / "assets"
if not (assets / "classifier.pt").exists() or not (assets / "validation_cases.npz").exists():
    raise FileNotFoundError("Demo assets are missing. Run: python demo/prepare_demo.py")

checkpoint = torch.load(assets / "classifier.pt", map_location="cpu", weights_only=True)
width = checkpoint["in_features"]
hidden = checkpoint["config"]["hidden_size"]
model = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, 1))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with np.load(assets / "validation_cases.npz", allow_pickle=False) as data:
    features = data["features"].copy()
    labels = data["lesion"].copy()
    tracers = data["tracer"].copy()
    case_names = [f"Scan {i + 1:03d}" for i in range(len(features))]

if features.ndim != 2 or features.shape[1] != width:
    raise ValueError("Demo feature dimensions do not match the checkpoint")
if not np.isfinite(features).all():
    raise ValueError("Demo features contain invalid values")

choices = case_names
case_indices = {name: i for i, name in enumerate(case_names)}
threshold = checkpoint["threshold"]


def predict(case_name):
    if case_name not in case_indices:
        raise gr.Error("Choose a scan from the list.")
    i = case_indices[case_name]
    inputs = torch.as_tensor(features[i:i + 1], dtype=torch.float32)
    inputs = (inputs - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    with torch.no_grad():
        probability = model(inputs).sigmoid().item()
    predicted = int(probability >= threshold)
    names = ["No tumor", "Tumor"]
    return (
        {"Tumor": probability, "No tumor": 1 - probability},
        names[predicted],
        names[int(labels[i])],
        "Correct" if predicted == labels[i] else "Incorrect",
    )


text_sizes = gr.themes.sizes.Size(
    xxs="11.7px", xs="13px", sm="15.6px", md="18.2px",
    lg="20.8px", xl="28.6px", xxl="33.8px",
)

theme = gr.themes.Base(
    primary_hue="gray", secondary_hue="gray", neutral_hue="gray", text_size=text_sizes
).set(
    body_background_fill="#e5e5e5",
    body_background_fill_dark="#111111",
    body_text_color="#111111",
    body_text_color_dark="#e5e5e5",
    block_background_fill="#f5f5f5",
    block_background_fill_dark="#202020",
    input_background_fill="#ffffff",
    input_background_fill_dark="#171717",
    border_color_primary="#c0c0c0",
    border_color_primary_dark="#525252",
    button_primary_background_fill="#171717",
    button_primary_background_fill_dark="#c0c0c0",
    button_primary_background_fill_hover="#404040",
    button_primary_background_fill_hover_dark="#e5e5e5",
    button_primary_text_color="#f5f5f5",
    button_primary_text_color_dark="#111111",
)


with gr.Blocks(title="Tumor Detection") as demo:
    gr.Markdown("# Tumor Detection")
    gr.Markdown(
        "These are validation scans from the AutoPET dataset. "
        "The demo runs the classifier on features already extracted from each PET/CT scan."
    )
    case = gr.Dropdown(choices=choices, value=case_names[0], label="Select a Scan")
    button = gr.Button("Predict", variant="primary")
    scores = gr.Label(label="Probabilities", num_top_classes=2)
    with gr.Row():
        prediction = gr.Textbox(label="Prediction", interactive=False)
        actual = gr.Textbox(label="Actual Result", interactive=False)
        comparison = gr.Textbox(label="Correct?", interactive=False)
    button.click(predict, inputs=case, outputs=[scores, prediction, actual, comparison], api_name="predict")


if __name__ == "__main__":
    demo.launch(server_name=os.getenv("DEMO_HOST", "127.0.0.1"),
                server_port=int(os.getenv("PORT", "7860")), share=False, footer_links=[], theme=theme)
