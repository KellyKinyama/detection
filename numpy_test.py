import torch
import numpy as np
import facenet_pytorch
import cv2
from PIL import Image
import faiss
# print(f"Torch version: {torch.__version__}")
# print(f"NumPy version: {np.__version__}")
# print(f"facenet_pytorch version: {facenet_pytorch.__version__}")
# print(f"OpenCV version: {cv2.__version__}")
# print(f"Pillow version: {Image.__version__}")
# print(f"FAISS version: {faiss.__version__}")

# Test tensor conversion
test_np_array = np.random.rand(10, 10).astype(np.float32)
try:
    test_tensor = torch.as_tensor(test_np_array)
    print("NumPy array to Torch tensor conversion successful.")
except RuntimeError as e:
    print(f"NumPy array to Torch tensor conversion FAILED: {e}")