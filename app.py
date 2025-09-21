%%writefile app.py
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rasterio
import albumentations as A
import os

# Define the UNet model architecture (copy the relevant classes from your notebook)
def conv_block(in_ch, out_ch, p_drop=0.2):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p_drop),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop):
      super().__init__()
      self.pool = nn.MaxPool2d(2)
      self.conv = conv_block(in_ch, out_ch, p_drop)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch, p_drop):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch1, in_ch1 // 2, 2, stride=2)
        self.conv = conv_block(in_ch2 + in_ch1 // 2, out_ch, p_drop)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        # Corrected padding
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        return self.conv(torch.cat([x2, x1], dim=1))

class BayesianUNet(nn.Module):
    def __init__(self, in_ch, base=32, p_drop=0.2):
        super().__init__()
        self.inc = conv_block(in_ch, base, p_drop)
        self.down1 = Down(base, base*2, p_drop)
        self.down2 = Down(base*2, base*4, p_drop)
        self.down3 = Down(base*4, base*8, p_drop)
        self.down4 = Down(base*8, base*8, p_drop)
        self.up1 = Up(base*8, base*8, base*4, p_drop)
        self.up2 = Up(base*4, base*4, base*2, p_drop)
        self.up3 = Up(base*2, base*2, base, p_drop)
        self.up4 = Up(base, base, base, p_drop)
        self.outc = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

# Load the trained model
@st.cache_resource
def load_model():
    model = BayesianUNet(in_ch=1) # Assuming single channel input
    model.load_state_dict(torch.load("bayesian_unet_model.pt", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Streamlit app title
st.title("Flood Detection using Bayesian UNet")

# File uploader
uploaded_file = st.file_uploader("Upload a SAR image (TIF)", type="tif")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_image.tif", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocess the image and perform inference (Implement this part)
    def preprocess_and_predict(image_path, model, size=256):
        with rasterio.open(image_path) as src:
            img = src.read().astype(np.float32)
        # Add preprocessing steps here similar to your dataset class
        # Example: normalization, clipping, resizing/cropping
        for c in range(img.shape[0]):
            ch = img[c]
            p1, p99 = np.percentile(ch, (1, 99))
            img[c] = np.clip(ch, p1, p99)
            img[c] = (img[c] - img[c].mean()) / (ch.std() + 1e-6)

        # Assuming single channel for now, adjust if multi-channel
        img = img[0, :, :] # Take the first channel

        # Apply the same transformations as during training/validation
        transform = A.Compose([
            A.CenterCrop(size, size, pad_if_needed=True),
        ])
        img_hwc = img[..., None]
        augmented = transform(image=img_hwc)
        img_aug = augmented['image'].transpose(2,0,1) # Channel first

        # Convert to tensor and add batch dimension
        input_tensor = torch.from_numpy(img_aug).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            # Enable MC Dropout for uncertainty estimation if needed
            # enable_mc_dropout(model)
            # For a single prediction without MC dropout uncertainty:
            logits = model(input_tensor)
            prediction = torch.sigmoid(logits).squeeze().cpu().numpy()

        return prediction #, uncertainty_map (if using MC dropout)


    prediction = preprocess_and_predict("temp_image.tif", model) #, uncertainty_map

    # Display the results
    st.image(uploaded_file, caption="Uploaded SAR Image", use_column_width=True)
    st.image(prediction, caption="Flood Prediction", use_column_width=True, cmap='viridis')
    # if uncertainty_map is not None:
    #     st.image(uncertainty_map, caption="Uncertainty Map", use_column_width=True, cmap='magma')

    # Clean up temporary file
    os.remove("temp_image.tif")
