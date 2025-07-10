import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np # Import numpy
import pickle # To save and load embeddings

# --- Configuration ---
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
workers = 0 if os.name == 'nt' else 4

# --- Initialize Models ---
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --- Define collate function ---
def collate_fn(x):
    return x[0]

# --- 1. Generate and Save Known Embeddings ---
def generate_and_save_embeddings(data_path, output_file='known_faces_embeddings.pkl'):
    print(f"\n--- Generating embeddings for known faces from '{data_path}' ---")
    dataset = datasets.ImageFolder(data_path)
    dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

    known_embeddings = {}
    
    for i, (x, y) in enumerate(loader):
        person_name = dataset.idx_to_class[y]
        print(f"Processing image {i+1}/{len(dataset.imgs)} for {person_name}...")
        
        x_aligned, prob = mtcnn(x, return_prob=True)
        if x_aligned is not None:
            embedding = resnet(x_aligned.unsqueeze(0)).detach().cpu().numpy()
            
            if person_name not in known_embeddings:
                known_embeddings[person_name] = []
            known_embeddings[person_name].append(embedding.flatten()) # Store as flattened numpy array
            print(f'  Face detected with probability: {prob:.6f}')
        else:
            print(f'  No face detected for {person_name} in this image.')

    # Average embeddings for each person if multiple images are available
    averaged_known_embeddings = {
        name: np.mean(np.array(embs), axis=0)
        for name, embs in known_embeddings.items()
    }

    with open(output_file, 'wb') as f:
        pickle.dump(averaged_known_embeddings, f)
    print(f"Embeddings saved to '{output_file}'")
    return averaged_known_embeddings

# Run this once to create your known faces database
# Make sure your 'data/known_faces' directory exists and contains subfolders for each person.
# Example: data/known_faces/John_Doe/pic1.jpg, data/known_faces/Jane_Doe/pic1.jpg
known_faces_directory = 'data/known_faces'
if not os.path.exists(known_faces_directory):
    print(f"Warning: '{known_faces_directory}' not found. Please create it and add known face images.")
    # Exit or handle this case as appropriate
    exit()

known_embeddings_db = generate_and_save_embeddings(known_faces_directory, 'known_faces_embeddings.pkl')