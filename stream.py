import tkinter as tk
from tkinter import filedialog, Button, Label, Frame
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import os
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt
import random

class GenderClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gender Classifier")
        self.root.geometry("800x600")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        
        self.setup_ui()
        
        self.camera = None
        self.is_camera_on = False
        
        self.current_image = None
        
    def load_model(self):
        model_path = os.path.join('model', 'model_0.pth')
        
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        
        model = torch.nn.Sequential()
        
        conv_block_1 = torch.nn.Sequential()
        conv_block_1.add_module('0', torch.nn.Conv2d(3, 32, kernel_size=3, padding=1))
        conv_block_1.add_module('1', torch.nn.ReLU())
        conv_block_1.add_module('2', torch.nn.Conv2d(32, 64, kernel_size=3, padding=1))
        conv_block_1.add_module('3', torch.nn.ReLU())
        conv_block_1.add_module('4', torch.nn.MaxPool2d(2))
        model.add_module('conv_block_1', conv_block_1)
        
        conv_block_2 = torch.nn.Sequential()
        conv_block_2.add_module('0', torch.nn.Conv2d(64, 128, kernel_size=3, padding=1))
        conv_block_2.add_module('1', torch.nn.ReLU())
        conv_block_2.add_module('2', torch.nn.Conv2d(128, 128, kernel_size=3, padding=1))
        conv_block_2.add_module('3', torch.nn.ReLU())
        conv_block_2.add_module('4', torch.nn.MaxPool2d(2))
        model.add_module('conv_block_2', conv_block_2)
        
        model.add_module('flatten', torch.nn.Flatten())
        
        classifier = torch.nn.Sequential()
        classifier.add_module('0', torch.nn.Dropout(0.5))
        classifier.add_module('1', torch.nn.Linear(128 * 56 * 56, 2))
        model.add_module('classifier', classifier)
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model
    
    def setup_ui(self):
        main_frame = Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.image_label = Label(main_frame, text="Image will be displayed here", 
                                 width=640, height=480, bg="lightgray")
        self.image_label.pack(pady=10)
        
        button_frame = Frame(main_frame)
        button_frame.pack(pady=10)
        
        self.camera_button = Button(button_frame, text="Take Picture", 
                                   command=self.toggle_camera, width=15)
        self.camera_button.grid(row=0, column=0, padx=5)
        
        upload_button = Button(button_frame, text="Upload Picture", 
                              command=self.upload_image, width=15)
        upload_button.grid(row=0, column=1, padx=5)
        
        classify_button = Button(button_frame, text="Classify Gender", 
                                command=self.classify_gender, width=15)
        classify_button.grid(row=0, column=2, padx=5)
        
        test_button = Button(button_frame, text="Run Test", 
                            command=self.run_test, width=15)
        test_button.grid(row=0, column=3, padx=5)
        
        self.result_label = Label(main_frame, text="Classification result will appear here", 
                                 font=("Arial", 14))
        self.result_label.pack(pady=10)
    
    def toggle_camera(self):
        if self.is_camera_on:
            self.is_camera_on = False
            self.camera_button.config(text="Take Picture")
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            self.image_label.config(text="Image will be displayed here")
        else:
            self.is_camera_on = True
            self.camera_button.config(text="Capture")
            self.camera = cv2.VideoCapture(0)
            self.update_camera()
    
    def update_camera(self):
        if self.is_camera_on:
            ret, frame = self.camera.read()
            if ret:
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_image = Image.fromarray(cv2image)
                
                display_image = self.current_image.resize((640, 480))
                photo = ImageTk.PhotoImage(image=display_image)
                
                self.image_label.config(image=photo)
                self.image_label.image = photo
                
                self.root.after(10, self.update_camera)
            else:
                self.toggle_camera()
    
    def upload_image(self):
        if self.is_camera_on:
            self.toggle_camera()
        
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
        
        if file_path:
            self.current_image = Image.open(file_path)
            
            display_image = self.current_image.resize((640, 480))
            photo = ImageTk.PhotoImage(image=display_image)
            
            self.image_label.config(image=photo)
            self.image_label.image = photo
    
    def classify_gender(self):
        if self.current_image is None:
            self.result_label.config(text="No image to classify!")
            return
        
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        input_tensor = preprocess(self.current_image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_batch)
            
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        
        if predicted_class == 0:
            result = "Female"
        else:
            result = "Male"
        
        confidence = probabilities[predicted_class].item() * 100
        
        self.result_label.config(text=f"Predicted Gender: {result} (Confidence: {confidence:.2f}%)")
    
    def get_random_test_samples(self, test_dataloader, num_samples=5):
        test_batch = next(iter(test_dataloader))
        test_images, test_labels = test_batch
        
        random_indices = random.sample(range(len(test_images)), min(num_samples, len(test_images)))
        
        random_images = test_images[random_indices]
        random_labels = test_labels[random_indices]
        
        return random_images, random_labels
    
    def run_test(self):
        test_window = tk.Toplevel(self.root)
        test_window.title("Test Results")
        test_window.geometry("800x600")
        
        label = Label(test_window, text="In a real implementation, this would show test results\n"
                                       "with images and predictions similar to the notebook example.")
        label.pack(pady=20)

if __name__ == "__main__":
    root = tk.Tk()
    app = GenderClassifierApp(root)
    root.mainloop()
