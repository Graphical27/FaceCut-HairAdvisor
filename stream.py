import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import timm
from pathlib import Path

from transformers import ViTForImageClassification, ViTFeatureExtractor

model_path = Path("model")
tokenizer = ViTFeatureExtractor.from_pretrained(model_path)
model = ViTForImageClassification.from_pretrained(model_path,return_dict = False)
model.eval()
st.title("Image Classification with Vision Transformer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB") 
    st.image(image, caption="Uploaded Image", use_container_width=True)

    preprocess = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.TrivialAugmentWide(31),
    transforms.ToTensor()
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        logits = output.logits  
        pred_probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(pred_probs, dim=1)

        
    class_names = ["man","woman"]
    predicted_label = class_names[predicted_class] 

    st.write(f"Predicted Class: **{predicted_label}**")
    st.write("Raw model output:", output)
    st.write("Softmax probabilities:", torch.nn.functional.softmax(output, dim=1))