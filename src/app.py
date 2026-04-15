import streamlit as st
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
import faiss
import json
import os

# =========================
# PATH FIX (VERY IMPORTANT)
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

INDEX_PATH = os.path.join(BASE_DIR, "outputs/faiss.index")
PATHS_FILE = os.path.join(BASE_DIR, "outputs/image_paths.npy")
MODEL_PATH = os.path.join(BASE_DIR, "outputs/model/fine_tuned.pth")
META_PATH = os.path.join(BASE_DIR, "outputs/metadata.json")

# =========================
# Load FAISS + Data
# =========================
@st.cache_resource
def load_data():
    index = faiss.read_index(INDEX_PATH)
    paths = np.load(PATHS_FILE)

    paths = np.array([
        os.path.join(BASE_DIR, "data", "cleaned", os.path.basename(p))
        for p in paths
    ])

    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return index, paths, metadata

index, paths, metadata = load_data()

# DEBUG - مؤقت عشان نشوف المسار
st.write("Sample path:", paths[0])

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier = torch.nn.Identity()

    state_dict = torch.load(MODEL_PATH, map_location=device)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    return model, device

model, device = load_model()

# =========================
# Transform
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# UI
# =========================
st.set_page_config(layout="wide")

st.title("🏺 Egyptian Artifacts Recognition")

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

    cols = st.columns(5)

    for i, idx in enumerate(I[0]):
        path = paths[idx]
        filename = os.path.basename(path)

        with cols[i]:
            st.image(Image.open(path), use_container_width=True)

            if filename in metadata:
                info = metadata[filename]

                st.markdown(f"""
                **Name:** {info.get('name', 'N/A')}  
                **Location:** {info.get('location', 'N/A')}  
                **Age:** {info.get('age', 'N/A')}  
                """)
            else:
                st.caption("No metadata")