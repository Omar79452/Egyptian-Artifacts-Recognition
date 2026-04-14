import streamlit as st
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
import faiss
import os

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

INDEX_PATH = os.path.join(BASE_DIR, "outputs/faiss.index")
PATHS_FILE = os.path.join(BASE_DIR, "outputs/image_paths.npy")
MODEL_PATH = os.path.join(BASE_DIR, "outputs/model/fine_tuned.pth")

# =========================
# LOAD DATA (CACHED)
# =========================
@st.cache_resource
def load_data():
    index = faiss.read_index(INDEX_PATH)
    paths = np.load(PATHS_FILE)
    return index, paths

index, paths = load_data()

# =========================
# LOAD MODEL (CACHED)
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
# TRANSFORM
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

uploaded = st.file_uploader("📸 Upload any artifact image", type=["jpg", "png", "jpeg"])

if uploaded:
    st.success("✅ Image uploaded successfully")

    # 🔥 أهم سطر (no filename dependency)
    img = Image.open(uploaded).convert("RGB")

    st.image(img, caption="Query Image", width=300)

    # =========================
    # FEATURE EXTRACTION
    # =========================
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(img_tensor).cpu().numpy().astype("float32")

    faiss.normalize_L2(feat)

    # =========================
    # SEARCH
    # =========================
    D, I = index.search(feat, 5)

    st.subheader("🔍 Top Matches")

    cols = st.columns(5)

    for i, idx in enumerate(I[0]):
        raw_path = paths[idx]

        # 🔥 FIX PATH (important for deployment)
        filename = os.path.basename(raw_path)
        path = os.path.join(BASE_DIR, "data/cleaned", filename)

        with cols[i]:
            if os.path.exists(path):
                st.image(Image.open(path), use_container_width=True)
            else:
                st.error("Image not found")

            st.caption(filename)