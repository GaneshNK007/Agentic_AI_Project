import os
import io
import numpy as np
from PIL import Image

# ✅ Configure Keras to use PyTorch backend
os.environ["KERAS_BACKEND"] = "torch"
import keras
import torch

# ==========================================
# 📂 LOAD METASTATIC TISSUE SCREENING (PCAM) MODEL
# PCam Benchmark: Axillary Sentinel Lymph Node Histopathology (Breast Cancer Metastasis)
# NOTE: PCam evaluates metastatic breast adenocarcinoma in lymph nodes, NOT intracranial/brain tissue.
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = [
    os.path.join(BASE_DIR, "pcam.keras"),
    os.path.join(BASE_DIR, "best_pcam_resnet50_phase1_metastatic.keras"),
    os.path.join(BASE_DIR, "..", "..", "pcam.keras"),
    os.path.join(BASE_DIR, "..", "..", "best_pcam_resnet50_phase1_metastatic.keras"),
    "pcam.keras",
    "best_pcam_resnet50_phase1_metastatic.keras",
]

model_file = None
for p in MODEL_PATHS:
    if os.path.exists(p):
        model_file = p
        break

if model_file is None:
    raise FileNotFoundError("Could not locate pcam.keras or best_pcam_resnet50_phase1_metastatic.keras")

print(f"Loading Metastatic Tissue Screening (PCam) model from: {model_file}")
model = keras.models.load_model(model_file)
print("Metastatic Tissue Screening (PCam) model loaded successfully!")

# Class labels
class_names = ["Normal Lymphoid Tissue", "Metastatic Carcinoma"]


# ==========================================
# 🔬 PREDICT METASTATIC TISSUE ON HISTOPATHOLOGY PATCHES
# ==========================================
def predict_cancer(file, normalize=True):
    try:
        # Handle file input (UploadFile, bytes, PIL Image, or file path)
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

        # Resize to input shape (96, 96) for PCam ResNet50
        image = image.resize((96, 96))
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

        # Expand batch dimension: (1, 96, 96, 3)
        input_tensor = np.expand_dims(img_array, axis=0)

        # Inference with no grad tracking
        with torch.no_grad():
            outputs = model(input_tensor)
            if hasattr(outputs, "detach"):
                prob_raw = float(outputs.detach().cpu().numpy()[0][0])
            else:
                prob_raw = float(np.array(outputs)[0][0])

        # PCam model outputs sigmoid probability of metastatic cancer (1: Tumor, 0: Normal)
        prob_cancer = max(0.0, min(1.0, prob_raw))
        prob_normal = 1.0 - prob_cancer

        if prob_cancer >= 0.5:
            predicted_label = "Metastatic Carcinoma"
            confidence = prob_cancer
            if confidence >= 0.75:
                risk = "High - Metastatic carcinoma cells detected in lymph node patch. Urgent oncology staging required."
            else:
                risk = "Medium / Borderline - Suspicious malignant epithelial cluster detected. Pathologist recut & IHC review recommended."
        else:
            predicted_label = "Normal Lymphoid Tissue"
            confidence = prob_normal
            if confidence >= 0.75:
                risk = "Low - No metastatic carcinoma cells detected (benign lymphoid architecture)."
            else:
                risk = "Uncertain / Low - Predominantly benign lymphoid features, secondary pathologist review advised."

        # FORMAL UNCERTAINTY QUANTIFICATION: Normalized Binary Predictive Shannon Entropy
        # H(p) = - [p * log2(p) + (1-p) * log2(1-p)] in [0, 1]
        eps = 1e-7
        p_c = np.clip(prob_cancer, eps, 1.0 - eps)
        entropy = - (p_c * np.log2(p_c) + (1.0 - p_c) * np.log2(1.0 - p_c))
        uncertainty = round(float(np.clip(entropy, 0.0, 1.0)), 4)

        return {
            "prediction": predicted_label,
            "confidence": round(confidence, 4),
            "uncertainty": uncertainty,
            "uncertainty_metric": "Normalized Shannon Entropy (H in [0, 1])",
            "calibration_status": "Uncalibrated raw sigmoid output (operational decision-boundary proxy)",
            "risk": risk,
            "raw_class": predicted_label.lower().replace(" ", "_"),
            "dataset_provenance": "PatchCamelyon (CAMELYON16 Breast Cancer Sentinel Lymph Node Histopathology)",
            "all_probabilities": {
                "Normal Lymphoid Tissue": round(prob_normal, 4),
                "Metastatic Carcinoma": round(prob_cancer, 4)
            }
        }

    except Exception as e:
        return {"error": str(e)}