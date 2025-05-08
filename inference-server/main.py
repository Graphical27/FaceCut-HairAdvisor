from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
from mclass import MultiTaskModel

app = FastAPI()
# Enable CORS so frontend can call
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Load model once at startup
state_dict = torch.load("model/main_model.pth", map_location="cpu")
model = MultiTaskModel()
model.load_state_dict(state_dict)
model.eval()

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
gender_classes = ["male", "female"]
shape_classes = ["Heart", "Oblong", "Oval", "Round", "Square"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded file bytes
    contents = await file.read()
    # Wrap bytes in a file-like object for PIL
    img = Image.open(BytesIO(contents)).convert("RGB")

    # Preprocess and run inference
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        g_logits, s_logits = model(x)
        gender = gender_classes[g_logits.argmax(1).item()]
        shape = shape_classes[s_logits.argmax(1).item()]

    return {"gender": gender, "shape": shape}
