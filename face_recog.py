import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from facenet_pytorch import MTCNN, InceptionResnetV1
import pandas as pd

# Determine if an NVIDIA GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Set number of workers for DataLoader
workers = 0 if os.name == 'nt' else 4

# Define MTCNN module
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

# Define Inception Resnet V1 module
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Define a dataset and data loader
def collate_fn(x):
    return x[0]

# Assuming 'data/test_images' is in the same directory as this script
dataset = datasets.ImageFolder('data/test_images')
dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

# Perform MTCNN facial detection
aligned = []
names = []
print("Performing face detection...")
for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        print(f'Face detected with probability: {prob:.6f}')
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])

if not aligned:
    print("No faces detected in the provided dataset.")
else:
    # Calculate image embeddings
    print("\nCalculating image embeddings...")
    aligned = torch.stack(aligned).to(device)
    embeddings = resnet(aligned).detach().cpu()

    # Print distance matrix for classes
    print("\nGenerating distance matrix...")
    dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
    df_dists = pd.DataFrame(dists, columns=names, index=names)
    print(df_dists)