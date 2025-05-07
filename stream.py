import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn

@st.cache_resource
def load_model():
    state_dict = torch.load("model/main_model.pth", map_location="cpu")
    from mclass import MultiTaskModel  
    m = MultiTaskModel()  
    m.load_state_dict(state_dict)
    m.eval()
    return m

model = load_model()

gender_classes = ["male","female"]
shape_classes = ['Heart', 'Oblong', 'Oval', 'Round', 'Sqaure'] 

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict(img: Image.Image):
    x = transform(img).unsqueeze(0)  
    with torch.no_grad():
        g_logits, s_logits = model(x)
        g = gender_classes[g_logits.argmax(1).item()]
        s = shape_classes[s_logits.argmax(1).item()]
    return g, s

st.title("🚀 Gender & Face‑Shape Classifier")

mode = st.radio("Input:", ["Upload an image","Use camera"])
img: Image.Image = None

if mode=="Upload an image":
    up = st.file_uploader("Choose JPG/PNG", type=["jpg","jpeg","png"])
    if up:
        img = Image.open(up).convert("RGB")
else:
    cam = st.camera_input("Take a selfie")
    if cam:
        img = Image.open(cam).convert("RGB")

if img:
    st.image(img, caption="Input", use_column_width=True)
    gender, shape = predict(img)
    st.write(f"**Predicted Gender:** {gender}")
    st.write(f"**Predicted Face Shape:** {shape}")
