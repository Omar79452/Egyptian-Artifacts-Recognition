import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

DATASET_PATH = "../data/cleaned"
FEATURES_PATH = "../outputs/features.npy"
PATHS_PATH = "../outputs/image_paths.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

model.classifier = torch.nn.Identity()

state_dict = torch.load("../outputs/model/fine_tuned.pth", map_location=device)
state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}

model.load_state_dict(state_dict, strict=False)

model = model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image_paths = sorted(os.listdir(DATASET_PATH))

features = []
valid_paths = []

print("Extracting features...")

for img_name in tqdm(image_paths):
    path = os.path.join(DATASET_PATH, img_name)

    try:
        img = Image.open(path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = model(img)

        features.append(feat.cpu().numpy().flatten())
        valid_paths.append(path)

    except:
        continue

features = np.array(features)

os.makedirs("../outputs", exist_ok=True)

np.save(FEATURES_PATH, features)
np.save(PATHS_PATH, np.array(valid_paths))

print("Features shape:", features.shape)
print("Saved successfully")