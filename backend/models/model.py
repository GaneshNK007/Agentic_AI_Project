# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# import torch.nn.functional as F

# # ✅ Device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ✅ Load model (only once)
# model = models.resnet50(weights=None)
# model.fc = nn.Sequential(
#     nn.Dropout(0.5),
#     nn.Linear(model.fc.in_features, 4)
# )

# model.load_state_dict(torch.load("backend/models/brain_tumor_model.pth", map_location=device))
# model = model.to(device)
# model.eval()

# # ✅ Classes
# class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# # ✅ Better normalization (important improvement)
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         [0.485, 0.456, 0.406],   # 🔥 FIXED
#         [0.229, 0.224, 0.225]    # 🔥 FIXED
#     )
# ])

# def predict_image(file):
#     try:
#         # ✅ Handle both UploadFile and normal file
#         if hasattr(file, "file"):
#             image = Image.open(file.file).convert("RGB")
#         else:
#             image = Image.open(file).convert("RGB")

#         # ✅ Transform
#         input_tensor = transform(image).unsqueeze(0).to(device)

#         # ✅ Inference (fast + safe)
#         with torch.no_grad():
#             outputs = model(input_tensor)
#             probs = F.softmax(outputs, dim=1)
#             confidence, predicted_class = torch.max(probs, 1)

#         confidence = float(confidence.item())
#         predicted_label = class_names[predicted_class.item()]

#         # ✅ Risk logic
#         if confidence < 0.60:
#             risk = "Uncertain — Expert review required"
#         elif predicted_label == "notumor":
#             risk = "No tumor detected"
#         else:
#             risk = "Tumor detected — Clinical consultation recommended"

#         return {
#             "prediction": predicted_label,
#             "confidence": round(confidence, 4),
#             "risk": risk
#         }

#     except Exception as e:
#         return {
#             "error": str(e)
#         }


import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 4)
)

model.load_state_dict(torch.load("backend/models/brain_tumor_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Classes
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =========================
# 🧠 MAIN FUNCTION
# =========================
def predict_brain(file):
    try:
        # Handle file input
        if hasattr(file, "file"):
            image = Image.open(file.file).convert("RGB")
        else:
            image = Image.open(file).convert("RGB")

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probs, 1)

        confidence = float(confidence.item())
        predicted_label = class_names[predicted_class.item()]

        # 🔥 Standardized prediction text
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

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "risk": risk,
            "raw_class": predicted_label   # 🔥 useful for debugging/UI
        }

    except Exception as e:
        return {"error": str(e)}