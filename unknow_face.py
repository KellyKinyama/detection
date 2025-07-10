import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import pickle
from PIL import Image # For loading individual images

# --- Configuration (same as before) ---
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
workers = 0 if os.name == 'nt' else 4

# --- Initialize Models (same as before) ---
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --- Load Known Embeddings ---
try:
    with open('known_faces_embeddings.pkl', 'rb') as f:
        known_embeddings_db = pickle.load(f)
    print("Known embeddings loaded successfully.")
except FileNotFoundError:
    print("Error: 'known_faces_embeddings.pkl' not found. Please run the enrollment script first.")
    exit()

# Convert known embeddings to a list of (name, embedding) tuples for easier iteration
known_names = list(known_embeddings_db.keys())
known_embeddings_list = [known_embeddings_db[name] for name in known_names]
known_embeddings_tensor = torch.tensor(np.array(known_embeddings_list), device=device) # Convert to tensor for batch processing

# --- Define Recognition Function ---
def recognize_face(image_path, known_names, known_embeddings_tensor, threshold=0.9):
    """
    Performs face detection and recognition on a single image.

    Args:
        image_path (str): Path to the image file.
        known_names (list): List of names corresponding to known embeddings.
        known_embeddings_tensor (torch.Tensor): Tensor of known face embeddings.
        threshold (float): Maximum Euclidean distance for a face to be considered 'known'.

    Returns:
        str: The recognized name or "Unknown".
        float: The confidence score (1 - normalized_distance if known, 0 if unknown).
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
    unknown_embedding = resnet(x_aligned.unsqueeze(0)).detach().cpu().numpy().flatten()
    unknown_embedding_tensor = torch.tensor(unknown_embedding, device=device)

    # Calculate distances to all known embeddings
    # Using L2 norm (Euclidean distance)
    # (unknown_embedding_tensor - known_embeddings_tensor) calculates element-wise difference
    # .norm(dim=1) calculates the L2 norm along dimension 1 (for each known embedding)
    distances = (unknown_embedding_tensor - known_embeddings_tensor).norm(dim=1).cpu().numpy()

    min_distance_idx = np.argmin(distances)
    min_distance = distances[min_distance_idx]
    
    recognized_name = "Unknown"
    confidence = 0.0

    if min_distance < threshold:
        recognized_name = known_names[min_distance_idx]
        # A simple way to get confidence from distance (closer is higher confidence)
        # You might need to calibrate this based on your dataset and threshold
        confidence = 1.0 - (min_distance / threshold) # Normalize distance and invert
        if confidence < 0: confidence = 0 # Ensure confidence is not negative
    
    print(f"  Minimum distance to known faces: {min_distance:.4f}")
    print(f"  Result: {recognized_name} (Confidence: {confidence:.2f})")
    
    return recognized_name, confidence

# --- Example Usage ---
# Create a dummy image for testing, or use a real path
# Make sure to place an image of a known person or an unknown person here.
unknown_face_image_path = 'data/unknown_faces/IMG-20250303-WA0006.jpg' 
# You'll need to create 'data/unknown_faces/' and put an image inside for testing.

if not os.path.exists('data/unknown_faces'):
    os.makedirs('data/unknown_faces')
    print("Created 'data/unknown_faces' directory. Please place an image there for testing.")
else:
    if os.path.exists(unknown_face_image_path):
        print(f"\n--- Recognizing face in '{unknown_face_image_path}' ---")
        recognized_person, score = recognize_face(unknown_face_image_path, known_names, known_embeddings_tensor, threshold=0.9)
        print(f"\nRecognized: {recognized_person} with score: {score:.2f}")
    else:
        print(f"\nNo image found at '{unknown_face_image_path}'. Please place an image there to test recognition.")

# --- Testing with a known face (optional) ---
# Assuming 'data/known_faces/angelina_jolie/001.jpg' exists from your setup
# known_test_image_path = 'data/known_faces/angelina_jolie/001.jpg' 
# if os.path.exists(known_test_image_path):
#     print(f"\n--- Recognizing known face in '{known_test_image_path}' ---")
#     recognized_person_known, score_known = recognize_face(known_test_image_path, known_names, known_embeddings_tensor, threshold=0.9)
#     print(f"\nRecognized (Known): {recognized_person_known} with score: {score_known:.2f}")