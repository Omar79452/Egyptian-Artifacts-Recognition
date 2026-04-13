import streamlit as st
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
import faiss
import json
import os

# =========================
# Load Data
# =========================
index = faiss.read_index("../outputs/faiss.index")
paths = np.load("../outputs/image_paths.npy")

# Load metadata
if os.path.exists("../outputs/metadata.json"):
    with open("../outputs/metadata.json") as f:
        metadata = json.load(f)
else:
    metadata = {}

# =========================
# Model
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier = torch.nn.Identity()

# Load fine-tuned
model.load_state_dict(torch.load("../outputs/model/fine_tuned.pth", map_location=device))

model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# UI
# =========================
st.title("Egyptian Artifacts Recognition 🔥")

uploaded = st.file_uploader("Upload an artifact image", type=["jpg", "png", "jpeg"])

if uploaded:
    st.success("✅ Image uploaded successfully")

    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Query Image", width=300)

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(img_tensor).cpu().numpy().astype("float32")

    faiss.normalize_L2(feat)

    D, I = index.search(feat, 5)

    st.subheader("🔍 Top Matches")

    for i, idx in enumerate(I[0]):
        path = paths[idx]
        filename = os.path.basename(path)

        result_img = Image.open(path)

        st.image(result_img, width=200)

        if filename in metadata:
            info = metadata[filename]

            st.markdown(f"""
            **Name:** {info['name']}  
            **Description:** {info['description']}  
            **Location:** {info['location']}  
            **Age:** {info['age']}  
            """)
        else:
            st.info("No metadata available")

        st.markdown("---")