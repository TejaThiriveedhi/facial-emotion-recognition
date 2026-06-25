# Facial Emotion Recognition with EfficientNetB0 & Grad-CAM

A deep learning system that classifies facial emotions from images or live webcam feed into 7 categories, with visual explainability via Grad-CAM++ heatmaps. Built on EfficientNetB0 with fine-tuning on the FER2013 dataset and deployed as an interactive Gradio web app.

![Sample Prediction](/sample_images.png)

---

## Demo

Upload a face image or use your webcam — the app predicts the emotion and overlays a Grad-CAM++ heatmap showing which facial regions drove the prediction.

![Grad-CAM Visualization](/gradcam_pp_efficientnet.png)

---

## Features

- **7-class emotion classification** — Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **EfficientNetB0 backbone** — pretrained on ImageNet, fine-tuned on FER2013
- **Grad-CAM++ explainability** — visual heatmap highlighting the regions influencing each prediction
- **Interactive Gradio app** — supports image upload and live webcam capture
- **Training diagnostics** — confusion matrix, per-class accuracy, and training curves included

---

## Model Architecture

- **Base model:** EfficientNetB0 (ImageNet weights, top removed)
- **Fine-tuning:** Last 30 layers unfrozen with a low learning rate
- **Head:** GlobalAveragePooling → BatchNorm → Dropout(0.4) → Dense(256, ReLU) → Dense(7, Softmax)
- **Dataset:** FER2013 (35,887 grayscale images, 48×48, converted to RGB)
- **Explainability:** Grad-CAM++ on the last convolutional block

---

## Results

| Metric | Value |
|---|---|
| Dataset | FER2013 |
| Backbone | EfficientNetB0 |
| Input size | 48×48 → 224×224 |

![Confusion Matrix](/confusion_matrix_eff.png)
![Training Curves](/training_curves.png)
![Class Distribution](/class_distribution.png)

---

## Project Structure

```
facial-emotion-recognition/
│
├── FER2013_Emotion_Detection.ipynb   # Training pipeline (Colab)
├── FER2013_Gradio_App.ipynb          # Gradio app notebook version
├── gradio_app.py                     # Standalone Gradio app script
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── efficientnetb0_single.keras   # Single-model weights
│
├── saved_models/
│   └── efficientnet_full.keras       # Full saved model (architecture + weights)
│
├── logs/
│   └── efficientnet_training_log.csv # Epoch-by-epoch training history
│
└── /                           # Visualizations used in README
    ├── sample_images.png
    ├── gradcam_pp_efficientnet.png
    ├── confusion_matrix_eff.png
    ├── training_curves.png
    └── class_distribution.png
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/facial-emotion-recognition.git
cd facial-emotion-recognition
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Gradio app

```bash
python gradio_app.py
```

Open the local URL shown in the terminal — upload a face image or use your webcam to get a prediction with Grad-CAM overlay.

---

## Training

Open `FER2013_Emotion_Detection.ipynb` in Google Colab. The notebook covers:

- Dataset loading and preprocessing from FER2013
- EfficientNetB0 feature extraction phase
- Fine-tuning phase with unfrozen layers
- Evaluation: confusion matrix, per-class accuracy, training curves
- Grad-CAM++ visualization on test samples

---

## Tech Stack

| Component | Tool |
|---|---|
| Deep learning framework | TensorFlow / Keras |
| Model backbone | EfficientNetB0 |
| Explainability | Grad-CAM++ |
| App framework | Gradio |
| Dataset | FER2013 (Kaggle) |
| Training environment | Google Colab |

---

