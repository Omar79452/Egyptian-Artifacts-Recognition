import torch
import os
from torchvision import models, transforms
from torch import nn, optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader

DATA_PATH = "../data/cleaned"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Custom Dataset (no labels)
# =====================
class ImageDataset(Dataset):
    def __init__(self, folder, transform):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder)]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.transform(img)  # نفس الصورة مرتين

# =====================
# Transform
# =====================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

dataset = ImageDataset(DATA_PATH, transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# =====================
# Model
# =====================
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# freeze everything
for param in model.parameters():
    param.requires_grad = False

# unfreeze last block
for param in model.features[-1].parameters():
    param.requires_grad = True

model = model.to(device)

# =====================
# Simple Loss (Similarity-based)
# =====================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# =====================
# Training Loop
# =====================
for epoch in range(3):
    total_loss = 0

    for x1, x2 in loader:
        x1, x2 = x1.to(device), x2.to(device)

        f1 = model(x1)
        f2 = model(x2)

        loss = criterion(f1, f2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "../outputs/model/fine_tuned.pth")
print("✅ Model saved")