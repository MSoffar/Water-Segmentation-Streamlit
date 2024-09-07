import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import os

# Set page configuration at the very beginning
st.set_page_config(
    page_title="Water Segmentation with U-Net 🌊",
    page_icon="💧",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Define your model architecture (same as the one used during training)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.enc1 = DoubleConv(n_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bridge = DoubleConv(512, 1024)
        self.dec4 = DoubleConv(1024, 512)
        self.dec3 = DoubleConv(512, 256)
        self.dec2 = DoubleConv(256, 128)
        self.dec1 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        bridge = self.bridge(self.pool(enc4))

        dec4 = self.dec4(self.center_crop_and_concat(self.up4(bridge), enc4))
        dec3 = self.dec3(self.center_crop_and_concat(self.up3(dec4), enc3))
        dec2 = self.dec2(self.center_crop_and_concat(self.up2(dec3), enc2))
        dec1 = self.dec1(self.center_crop_and_concat(self.up1(dec2), enc1))

        return self.final(dec1)

    def center_crop_and_concat(self, upsampled, bypass):
        crop_size = (bypass.size(2) - upsampled.size(2)) // 2
        if crop_size > 0:
            bypass = bypass[:, :, crop_size:-crop_size, crop_size:-crop_size]
        return torch.cat([upsampled, bypass], dim=1)

# Initialize and load the model
model = UNet(n_channels=12, n_classes=1)
model_path = 'model.pth'

try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    st.error(f"Failed to load the model: {e}")

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32)  # Read the image as a NumPy array
    image = torch.from_numpy(image).unsqueeze(0)  # Add batch dimension
    return image

# Streamlit app with enhanced UI
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>Water Segmentation with U-Net 🌊</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #8E44AD;'>Upload your TIFF image to see the magic! ✨</h3>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a TIFF image... 📂", type=["tif"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp.tif", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.markdown("<h4 style='color: #2980B9;'>Processing your image... please wait! ⏳</h4>", unsafe_allow_html=True)
    with st.spinner("Segmenting... 🔄"):
        input_tensor = preprocess_image("temp.tif")

        with torch.no_grad():
            output = model(input_tensor)
            prediction = (torch.sigmoid(output) > 0.5).float().cpu().numpy()

    st.success("Prediction Complete! 🎉")
    st.markdown("<h4 style='color: #27AE60;'>Here is your segmented mask:</h4>", unsafe_allow_html=True)

    # Plot the original image and the segmentation result side by side
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Display the first channel of the input image (adjust if necessary)
    ax[0].imshow(input_tensor[0, 0].cpu().numpy(), cmap='gray')
    ax[0].set_title('Original Image (Channel 1)', fontsize=15)

    # Display the predicted segmentation mask
    ax[1].imshow(prediction[0, 0], cmap='gray')
    ax[1].set_title('Predicted Segmentation Mask', fontsize=15)

    for a in ax:
        a.axis('off')

    st.pyplot(fig)
    st.balloons()

# Sidebar for additional information
st.sidebar.markdown("### About the App 💡")
st.sidebar.write("This app uses a U-Net model to segment water bodies in satellite images. "
                 "Simply upload a TIFF image, and the model will predict the water regions. "
                 "It's an intuitive tool for environmental analysis, urban planning, and more!")

st.sidebar.markdown("### Tips for Best Results 📝")
st.sidebar.write("1. Ensure your TIFF image has 12 channels.\n"
                 "2. Use clear, high-resolution images for more accurate segmentation.\n"
                 "3. Explore different channels in the original image to understand the data better.")
