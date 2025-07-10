import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import pickle
from PIL import Image
import faiss # Import FAISS

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

# --- 1. Generate and Save Known Embeddings with FAISS Index ---
def generate_and_save_faiss_index(data_path, index_file='face_recognition_index.faiss', names_file='face_names.pkl'):
    print(f"\n--- Generating embeddings and FAISS index for known faces from '{data_path}' ---")
    dataset = datasets.ImageFolder(data_path)
    dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

    all_embeddings = []
    all_names = []
    
    # Temporarily store embeddings per person to average them
    temp_embeddings_per_person = {}

    for i, (x, y) in enumerate(loader):
        person_name = dataset.idx_to_class[y]
        
        x_aligned, prob = mtcnn(x, return_prob=True)
        if x_aligned is not None:
            embedding = resnet(x_aligned.unsqueeze(0)).detach().cpu().numpy().flatten()
            
            if person_name not in temp_embeddings_per_person:
                temp_embeddings_per_person[person_name] = []
            temp_embeddings_per_person[person_name].append(embedding)
        # else:
            # print(f'  No face detected for {person_name} in image {i}.') # Can be noisy for large datasets

    # Average embeddings for each person and prepare for FAISS
    current_idx = 0
    name_to_idx_map = {} # To map names to their index in the FAISS index
    idx_to_name_map = {} # To map FAISS index back to names

    for name, embeddings_list in temp_embeddings_per_person.items():
        if embeddings_list:
            avg_embedding = np.mean(np.array(embeddings_list), axis=0)
            all_embeddings.append(avg_embedding)
            all_names.append(name) # Store the actual name
            
            # Map the name to the current index
            name_to_idx_map[name] = current_idx
            idx_to_name_map[current_idx] = name
            current_idx += 1
    
    if not all_embeddings:
        print("No embeddings generated. Please check your known faces directory.")
        return None, None

    embeddings_np = np.array(all_embeddings).astype('float32')
    embedding_dim = embeddings_np.shape[1]

    # Initialize a FAISS index (e.g., IndexFlatL2 for brute-force L2 distance)
    # For truly large scale (millions+), you'd use IndexIVFFlat, IndexHNSW, etc.
    index = faiss.IndexFlatL2(embedding_dim) 
    print(f"Adding {len(embeddings_np)} embeddings to FAISS index...")
    index.add(embeddings_np)
    print(f"FAISS index built with {index.ntotal} vectors.")

    faiss.write_index(index, index_file)
    with open(names_file, 'wb') as f:
        pickle.dump(idx_to_name_map, f)
    
    print(f"FAISS index saved to '{index_file}' and names map to '{names_file}'")
    return index, idx_to_name_map

# --- 2. Perform Recognition on an Unknown Face using FAISS ---
def recognize_face_large_scale(image_path, faiss_index, idx_to_name_map, threshold=0.9):
    """
    Performs face detection and recognition on a single image using a FAISS index.
    """
    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return "Error: Image not found", 0.0
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return "Error: Image loading failed", 0.0

    # Detect face
    x_aligned, prob = mtcnn(img, return_prob=True)

    if x_aligned is None:
        print(f"No face detected in {image_path}")
        return "Unknown", 0.0
    
    print(f"Face detected in {image_path} with probability: {prob:.6f}")

    # Calculate embedding for the unknown face
    unknown_embedding = resnet(x_aligned.unsqueeze(0)).detach().cpu().numpy().astype('float32')

    # Search in FAISS index (k=1 for nearest neighbor)
    distances, indices = faiss_index.search(unknown_embedding, 1) # Find 1 nearest neighbor
    
    min_distance = distances[0][0] # The distance to the closest match
    closest_index = indices[0][0] # The index of the closest match in the FAISS index

    recognized_name = "Unknown"
    confidence = 0.0

    if min_distance < threshold:
        recognized_name = idx_to_name_map.get(closest_index, "Unknown (Index Error)")
        confidence = 1.0 - (min_distance / threshold)
        if confidence < 0: confidence = 0
    
    print(f"  Minimum distance to known faces: {min_distance:.4f}")
    print(f"  Result: {recognized_name} (Confidence: {confidence:.2f})")
    
    return recognized_name, confidence

# --- Example Usage ---
known_faces_directory = 'data/known_faces'
faiss_index_file = 'face_recognition_index.faiss'
names_map_file = 'face_names.pkl'

# Ensure known faces directory exists
if not os.path.exists(known_faces_directory):
    print(f"Warning: '{known_faces_directory}' not found. Please create it and add known face images.")
    exit()

# Generate and save the FAISS index (run this once to populate your database)
faiss_index, idx_to_name_map = generate_and_save_faiss_index(
    known_faces_directory, faiss_index_file, names_map_file
)

if faiss_index is None: # Exit if no embeddings were generated
    exit()

# Load the FAISS index and names if not already loaded (e.g., if running this script multiple times)
if faiss_index is None or idx_to_name_map is None:
    try:
        faiss_index = faiss.read_index(faiss_index_file)
        with open(names_map_file, 'rb') as f:
            idx_to_name_map = pickle.load(f)
        print("FAISS index and names loaded successfully.")
    except FileNotFoundError:
        print("Error: FAISS index or names file not found. Please run the enrollment script first.")
        exit()

# Prepare an unknown face image for recognition
unknown_face_image_path = 'data/unknown_faces/IMG-20250303-WA0006.jpg' 

if not os.path.exists('data/unknown_faces'):
    os.makedirs('data/unknown_faces')
    print("Created 'data/unknown_faces' directory. Please place an image there for testing.")
else:
    if os.path.exists(unknown_face_image_path):
        print(f"\n--- Recognizing face in '{unknown_face_image_path}' using FAISS ---")
        recognized_person, score = recognize_face_large_scale(
            unknown_face_image_path, faiss_index, idx_to_name_map, threshold=0.9
        )
        print(f"\nRecognized: {recognized_person} with score: {score:.2f}")
    else:
        print(f"\nNo image found at '{unknown_face_image_path}'. Please place an image there to test recognition.")