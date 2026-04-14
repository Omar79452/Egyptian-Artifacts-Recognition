import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
import faiss
import os
import matplotlib.pyplot as plt
import wikipedia
import sys

# =========================
# CONFIG
# =========================
INDEX_PATH = "../outputs/faiss.index"
PATHS_FILE = "../outputs/image_paths.npy"
MODEL_PATH = "../outputs/model/fine_tuned.pth"
TOP_K = 5

# =========================
# Wikipedia Setup
# =========================
wikipedia.set_lang("en")

def get_info(name):
    try:
        name = name.replace("_", " ").replace("-", " ")
        summary = wikipedia.summary(name, sentences=2)
        page = wikipedia.page(name)
        return page.title, summary, page.url
    except:
        return None, None, None

# =========================
# START CLEAN
# =========================
plt.close('all')

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load Model
# =========================
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier = torch.nn.Identity()

state_dict = torch.load(MODEL_PATH, map_location=device)
state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}
model.load_state_dict(state_dict, strict=False)

model = model.to(device)
model.eval()

# =========================
# Transform
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# Load FAISS
# =========================
if not os.path.exists(INDEX_PATH):
    print("❌ FAISS index not found. Run 4_build_index.py first.")
    exit()

index = faiss.read_index(INDEX_PATH)
paths = np.load(PATHS_FILE)

# =========================
# Load Query
# =========================
QUERY_PATH = sys.argv[1] if len(sys.argv) > 1 else None

if QUERY_PATH is None or not os.path.exists(QUERY_PATH):
    print("❌ Query image not found! Pass image path as argument.")
    exit()

query_img = Image.open(QUERY_PATH).convert("RGB")
img_tensor = transform(query_img).unsqueeze(0).to(device)

# =========================
# Extract Feature
# =========================
with torch.no_grad():
    query_feat = model(img_tensor).cpu().numpy().astype("float32")

faiss.normalize_L2(query_feat)

# =========================
# Search
# =========================
distances, indices = index.search(query_feat, TOP_K)

print("\n🔍 Top Matches:\n")

results = []
for idx in indices[0]:
    print(paths[idx])
    results.append(paths[idx])

# =========================
# SHOW
# =========================
fig, axes = plt.subplots(1, TOP_K + 1, figsize=(25, 8))

axes[0].imshow(query_img)
axes[0].set_title("Query")
axes[0].axis("off")

for i, path in enumerate(results):
    img = Image.open(path)
    filename = os.path.basename(path).split(".")[0]

    title, summary, url = get_info(filename)

    axes[i + 1].imshow(img)
    axes[i + 1].axis("off")

    if title:
        axes[i + 1].set_title(title[:25])
    else:
        axes[i + 1].set_title("Match")

    if title:
        print(f"\n📚 {title}")
        print(summary)
        print(url)

plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.05, wspace=0.2)

mng = plt.get_current_fig_manager()
try:
    mng.window.state('zoomed')
except:
    try:
        mng.window.showMaximized()
    except:
        pass

plt.show()