import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os

# Define the CNN model (Assuming you are using a pre-defined ResNet or your own CNN model)
class GenderClassificationModel(nn.Module):
    def __init__(self):
        super(GenderClassificationModel, self).__init__()
        # Add your model layers here
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Assuming 2 classes (Man and Woman)

    def forward(self, x):
        return self.model(x)

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate model and load state
model = GenderClassificationModel().to(device)

# Ensure loading the model from the correct directory
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'tt.pth')  # Updated model filename

# Load the saved model with strict=False to ignore missing keys
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()

# Define transformations for the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to predict the gender from image
def predict_gender(image, model):
    # Transform the image
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    
    # Forward pass through the model
    with torch.no_grad():
        output = model(image)
    
    # Get the predicted label
    _, predicted = torch.max(output, 1)
    return 'Man' if predicted.item() == 0 else 'Woman'

# Streamlit app interface
st.title("Gender Classification")

# Choose between real-time webcam or image upload
option = st.selectbox('Choose input method:', ('Upload Image', 'Real-time Webcam'))

prediction = None

if option == 'Upload Image':
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Load the uploaded image and display it
        pil_image = Image.open(uploaded_image)
        st.image(pil_image, caption='Uploaded Image', use_column_width=True)
        
        # Predict the gender from the uploaded image
        prediction = predict_gender(pil_image, model)

elif option == 'Real-time Webcam':
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame")
            break
        
        # Convert the image to RGB (OpenCV uses BGR by default)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the captured image in the Streamlit app
        FRAME_WINDOW.image(img_rgb)

        # Convert image to PIL format for model inference
        pil_image = Image.fromarray(img_rgb)

        # Predict the gender once for each frame and display the result
        if prediction is None:  # Ensure prediction happens only once per frame
            prediction = predict_gender(pil_image, model)

        # Display the prediction
        st.write(f"Prediction: {prediction}")
        
        # Reset prediction for next frame
        prediction = None

    cap.release()

# Display the prediction result for the uploaded image (if applicable)
if prediction is not None:
    st.write(f"Prediction: {prediction}")
