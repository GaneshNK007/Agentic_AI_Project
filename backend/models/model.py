import os
import io
import numpy as np
from PIL import Image

# ✅ Configure Keras to use PyTorch backend
os.environ["KERAS_BACKEND"] = "torch"
import keras
import torch

# =========================
# 📂 LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = [
    os.path.join(BASE_DIR, "best_resnet50_finetuned.keras"),
    os.path.join(BASE_DIR, "..", "..", "best_resnet50_finetuned.keras"),
    "best_resnet50_finetuned.keras",
]

model_file = None
for p in MODEL_PATHS:
    if os.path.exists(p):
        model_file = p
        break

if model_file is None:
    raise FileNotFoundError("Could not locate best_resnet50_finetuned.keras")

print(f"Loading Brain Tumor model from: {model_file}")
model = keras.models.load_model(model_file)
print("Brain Tumor model loaded successfully!")

# Classes
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# =========================
# 🧠 MAIN FUNCTION
# =========================
def predict_brain(file, normalize=True):
    try:
        # Handle file input (UploadFile, bytes, PIL Image, or path)
        if hasattr(file, "file"):
            file.file.seek(0)
            image_bytes = file.file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        elif isinstance(file, (bytes, bytearray)):
            image = Image.open(io.BytesIO(file)).convert("RGB")
        elif isinstance(file, Image.Image):
            image = file.convert("RGB")
        else:
            image = Image.open(file).convert("RGB")

        # Resize to input shape (224, 224)
        image = image.resize((224, 224))
        img_array = np.array(image, dtype=np.float32)

        # Preprocessing
        if normalize:
            # Standard ImageNet normalization: (x/255.0 - mean) / std
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_array = ((img_array / 255.0) - mean) / std
        else:
            # Standard [0, 1] scaling
            img_array = img_array / 255.0

        # Expand batch dimension: (1, 224, 224, 3)
        input_tensor = np.expand_dims(img_array, axis=0)

        # Inference with no grad tracking
        with torch.no_grad():
            outputs = model(input_tensor)
            if hasattr(outputs, "detach"):
                probs = outputs.detach().cpu().numpy()[0]
            else:
                probs = np.array(outputs)[0]

        predicted_class_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_class_idx])
        predicted_label = class_names[predicted_class_idx]

        # Standardized prediction text
        if predicted_label == "notumor":
            prediction = "No Tumor"
        else:
            prediction = f"{predicted_label.capitalize()} Tumor"

        # Risk logic
        if confidence < 0.60:
            risk = "Uncertain — Expert review required"
        elif predicted_label == "notumor":
            risk = "No tumor detected"
        else:
            risk = "Tumor detected — Clinical consultation recommended"

        # FORMAL UNCERTAINTY QUANTIFICATION: Normalized Multi-Class Shannon Entropy
        # H(p) = - sum(p_k * log_4(p_k)) in [0, 1]
        eps = 1e-7
        probs_clipped = np.clip(probs, eps, 1.0)
        probs_clipped = probs_clipped / np.sum(probs_clipped)
        entropy_base4 = - np.sum(probs_clipped * (np.log(probs_clipped) / np.log(len(class_names))))
        uncertainty = round(float(np.clip(entropy_base4, 0.0, 1.0)), 4)

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "uncertainty": uncertainty,
            "uncertainty_metric": "Normalized Shannon Entropy (Base-4, H in [0, 1])",
            "calibration_status": "Uncalibrated raw softmax output (operational decision-boundary proxy)",
            "risk": risk,
            "raw_class": predicted_label,
            "all_probabilities": {
                class_names[i]: round(float(probs[i]), 4)
                for i in range(len(class_names))
            }
        }

    except Exception as e:
        return {"error": str(e)}