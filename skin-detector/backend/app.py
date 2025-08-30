import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn.functional as F

# === Build Model ===
def build_model(num_classes: int):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Freeze semua layer
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze layer3 & layer4
    for p in model.layer3.parameters():
        p.requires_grad = True
    for p in model.layer4.parameters():
        p.requires_grad = True

    # Custom FC
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.35),
        nn.Linear(128, num_classes)
    )
    return model

# === Load Model ===
model_path = "Fix_best_model.pth"
model = None

try:
    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict):
        model = build_model(num_classes=23)
        model.load_state_dict(ckpt)
        print("✅ Model berhasil dimuat dari state_dict.")
    else:
        model = ckpt
        print("✅ Model berhasil dimuat sebagai full model.")
except Exception as e:
    print("❌ Gagal memuat model:", e)

if model:
    model.eval()

# === FastAPI App ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Skin Detector API aktif 🚀"}

# === Transformasi Gambar ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Class Labels Sesuai Dataset ===
class_labels = [
    "Acne and Rosacea",
    "Actinic Keratosis, Basal Cell Carcinoma & other Malignant Lesions",
    "Atopic Dermatitis",
    "Bullous Disease",
    "Cellulitis, Impetigo & other Bacterial Infections",
    "Eczema",
    "Exanthems & Drug Eruptions",
    "Hair Loss, Alopecia & other Hair Diseases",
    "Herpes, HPV & other STDs",
    "Light Diseases & Disorders of Pigmentation",
    "Lupus & other Connective Tissue diseases",
    "Melanoma, Skin Cancer, Nevi & Moles",
    "Nail Fungus & other Nail Disease",
    "Poison Ivy & other Contact Dermatitis",
    "Psoriasis, Lichen Planus & related diseases",
    "Scabies, Lyme Disease & other Infestations & Bites",
    "Seborrheic Keratoses & other Benign Tumors",
    "Systemic Disease",
    "Tinea, Ringworm, Candidiasis & other Fungal Infections",
    "Urticaria / Hives",
    "Vascular Tumors",
    "Vasculitis",
    "Warts, Molluscum & other Viral Infections"
]

# === Endpoint Prediksi ===
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            class_idx = predicted.item()
            confidence = confidence.item() * 100

        result = class_labels[class_idx]

        return {
            "predicted_class": result,
            "confidence": round(confidence, 2)  # persentase keyakinan
        }

    except Exception as e:
        return {"error": str(e)}
