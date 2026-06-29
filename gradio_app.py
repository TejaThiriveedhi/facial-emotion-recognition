
import os
import numpy as np
import tensorflow as tf
import cv2
import gradio as gr
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

BASE = "os.path.dirname(os.path.abspath(__file__))"
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_EMOJIS = {
    "angry": "😠", "disgust": "🤢", "fear": "😨",
    "happy": "😊", "neutral": "😐", "sad": "😢", "surprise": "😲"
}

def build_mobilenet():
    base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    base.trainable = True
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(7, activation="softmax")
    ])
    model.build((None, 224, 224, 3))
    return model

def build_efficientnet():
    base = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights=None)
    base.trainable = True
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(7, activation="softmax")
    ])
    model.build((None, 224, 224, 3))
    return model

mob_model = build_mobilenet()
mob_model.load_weights(f"{BASE}/checkpoints/mobilenet/mob_epoch_final.weights.h5")

eff_model = build_efficientnet()
eff_model.load_weights(f"{BASE}/checkpoints/efficientnet/eff_epoch_final.weights.h5")

def preprocess_mobilenet(img_rgb):
    img = cv2.resize(img_rgb, (224, 224))
    return img.astype(np.float32) / 255.0

def preprocess_efficientnet(img_rgb):
    img = cv2.resize(img_rgb, (224, 224))
    return tf.keras.applications.efficientnet.preprocess_input(img.astype(np.float32))

def make_gradcam_pp(model, img_array, emotion_idx):
    img_batch = tf.cast(np.expand_dims(img_array, axis=0), tf.float32)
    base_model = model.layers[0]
    last_conv_layer = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    conv_model = tf.keras.Model(inputs=base_model.input, outputs=last_conv_layer.output)
    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape0:
                conv_outputs = conv_model(img_batch)
                tape0.watch(conv_outputs)
                tape1.watch(conv_outputs)
                tape2.watch(conv_outputs)
                x = conv_outputs
                for layer in model.layers[1:]:
                    x = layer(x)
                loss = x[:, emotion_idx]
            grads = tape0.gradient(loss, conv_outputs)
        grads2 = tape1.gradient(grads, conv_outputs)
    grads3 = tape2.gradient(grads2, conv_outputs)
    grads = grads.numpy()[0]
    grads2 = grads2.numpy()[0]
    grads3 = grads3.numpy()[0]
    conv_out = conv_outputs.numpy()[0]
    global_sum = np.sum(conv_out, axis=(0, 1))
    alpha_num = grads2
    alpha_denom = 2.0 * grads2 + global_sum[np.newaxis, np.newaxis, :] * grads3
    alpha_denom = np.where(alpha_denom == 0, 1e-10, alpha_denom)
    alphas = alpha_num / alpha_denom
    weights = np.sum(alphas * np.maximum(grads, 0), axis=(0, 1))
    cam = np.zeros(conv_out.shape[:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_out[:, :, i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam

def overlay_heatmap(img_rgb, heatmap, alpha=0.4):
    img_resized = cv2.resize(img_rgb, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlaid = (1 - alpha) * img_resized.astype(np.float32) + alpha * heatmap_colored.astype(np.float32)
    return np.clip(overlaid, 0, 255).astype(np.uint8)

def analyze_emotion(image, model_choice):
    if image is None:
        return None, None, "Please upload an image."
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    img_rgb = image.copy()
    if model_choice == "MobileNetV2":
        model = mob_model
        img_preprocessed = preprocess_mobilenet(img_rgb)
    else:
        model = eff_model
        img_preprocessed = preprocess_efficientnet(img_rgb)
    preds = model.predict(np.expand_dims(img_preprocessed, axis=0), verbose=0)[0]
    predicted_idx = np.argmax(preds)
    predicted_emotion = EMOTIONS[predicted_idx]
    confidence = preds[predicted_idx] * 100
    heatmap = make_gradcam_pp(model, img_preprocessed, predicted_idx)
    gradcam_img = overlay_heatmap(img_rgb, heatmap)
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#e74c3c" if i == predicted_idx else "#3498db" for i in range(7)]
    emotion_labels = [f"{EMOTION_EMOJIS[e]} {e}" for e in EMOTIONS]
    bars = ax.barh(emotion_labels, preds * 100, color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)", fontsize=11)
    ax.set_title(f"Emotion Confidence Scores — {model_choice}", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, preds * 100):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    result_text = (
        f"Predicted Emotion: {EMOTION_EMOJIS[predicted_emotion]} {predicted_emotion.upper()}\n"
        f"Confidence: {confidence:.1f}%\n"
        f"Model: {model_choice}\n\n"
        f"All scores:\n" +
        "\n".join([f"  {EMOTION_EMOJIS[e]} {e:<10} {preds[i]*100:.1f}%"
                    for i, e in enumerate(EMOTIONS)])
    )
    return gradcam_img, fig, result_text

with gr.Blocks(title="Facial Emotion Recognition", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 😊 Facial Emotion Recognition System
    ### MobileNetV2 vs EfficientNetB0 with Grad-CAM++ Visualization
    Upload a face image to detect one of 7 emotions: **Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise**
    """)
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Face Image", type="numpy", height=300)
            model_choice = gr.Radio(choices=["MobileNetV2", "EfficientNetB0"],
                                    value="EfficientNetB0", label="Choose Model")
            analyze_btn = gr.Button("🔍 Analyze Emotion", variant="primary", size="lg")
        with gr.Column(scale=1):
            gradcam_output = gr.Image(label="Grad-CAM++ Attention Map", height=300)
    with gr.Row():
        with gr.Column(scale=2):
            chart_output = gr.Plot(label="Confidence Scores")
        with gr.Column(scale=1):
            text_output = gr.Textbox(label="Prediction Results", lines=12)
    analyze_btn.click(fn=analyze_emotion, inputs=[input_image, model_choice],
                      outputs=[gradcam_output, chart_output, text_output])
    gr.Markdown("""
    ---
    **How it works:**
    - 🧠 Both models trained on FER2013 dataset (35,887 images)
    - 🔥 Grad-CAM++ highlights which facial regions influenced the prediction
    - 📊 MobileNetV2: 59.6% accuracy | EfficientNetB0: 60.1% accuracy
    """)

demo.launch()
