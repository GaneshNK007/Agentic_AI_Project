import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# MODEL (2 OUTPUTS)
# =========================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)   # ✅ FIXED

model.load_state_dict(torch.load("backend/models/cancer_model.pth", map_location=device))
model = model.to(device)
model.eval()

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
])

# =========================
# PREDICT
# =========================
def predict_cancer(file):
    try:
        if hasattr(file, "file"):
            image_bytes = file.file.read()
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            img = Image.open(file).convert("RGB")

        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)

            # ✅ SOFTMAX (not sigmoid)
            probs = F.softmax(outputs, dim=1)

            confidence, predicted_class = torch.max(probs, 1)

        confidence = float(confidence.item())
        predicted_class = predicted_class.item()

        # ✅ Class mapping
        if predicted_class == 1:
            prediction = "Metastatic Cancer"
        else:
            prediction = "Normal Tissue"

        # Risk
        if confidence > 0.8:
            risk = "High"
        elif confidence > 0.6:
            risk = "Medium"
        else:
            risk = "Low"

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "risk": risk
        }

    except Exception as e:
        return {"error": str(e)}