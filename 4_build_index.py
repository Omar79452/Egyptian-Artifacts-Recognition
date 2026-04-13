import numpy as np
import faiss
import os

FEATURES_PATH = "../outputs/features.npy"
INDEX_PATH = "../outputs/faiss.index"

# Load features
features = np.load(FEATURES_PATH).astype("float32")

# Normalize (important for cosine similarity)
faiss.normalize_L2(features)

# Build index
dimension = features.shape[1]

index = faiss.IndexFlatIP(dimension)  # cosine similarity
index.add(features)

# Save index
faiss.write_index(index, INDEX_PATH)

print("✅ FAISS index built and saved")
print("Total vectors:", index.ntotal)