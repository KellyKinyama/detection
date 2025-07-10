import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import pickle
from PIL import Image
import faiss # Import FAISS
import cv2   # Import OpenCV

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
    # This collate function expects a tuple (image, label) from ImageFolder
    # and returns just the image (PIL Image).
    # If the DataLoader provides a list of (image, label) tuples,
    # and your collate_fn only takes x[0], it might break if x isn't a list/tuple.
    # Let's ensure it handles batched input correctly, although for single image processing,
    # the original collate_fn might be fine.
    # For robust image handling, let's assume x is a list of (img, label) tuples.
    # We only care about the image for MTCNN.
    return x[0] # This will return the PIL Image object

# --- 1. Generate and Save Known Embeddings with FAISS Index ---
def generate_and_save_faiss_index(data_path, index_file='face_recognition_index.faiss', names_file='face_names.pkl'):
    print(f"\n--- Generating embeddings and FAISS index for known faces from '{data_path}' ---")
    dataset = datasets.ImageFolder(data_path)
    dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
    
    all_embeddings = []
    all_names = []
    temp_embeddings_per_person = {}

    # Iterate directly over the dataset to get (PIL Image, label) pairs.
    for i, (pil_img, y_label) in enumerate(dataset):
        person_name = dataset.idx_to_class[y_label]
        
        # --- THIS IS THE FIX ---
        # Pass the PIL image directly to mtcnn.detect. Do NOT convert to a numpy array first.
        boxes, probs = mtcnn.detect(pil_img)
        
        if boxes is None:
            print(f'  No face detected for {person_name} in image {i+1}. Skipping.')
            continue
        
        found_faces_for_this_image = False
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(b) for b in box]
            
            face_img_pil = pil_img.crop((x1, y1, x2, y2))
            
            x_aligned = mtcnn(face_img_pil)
            
            if x_aligned is not None:
                embedding = resnet(x_aligned.unsqueeze(0).to(device)).detach().cpu().numpy().flatten()
                
                if person_name not in temp_embeddings_per_person:
                    temp_embeddings_per_person[person_name] = []
                temp_embeddings_per_person[person_name].append(embedding)
                found_faces_for_this_image = True
            else:
                print(f'  Could not get aligned face {j+1} for {person_name} in image {i+1}. Skipping this face.')
        
        if not found_faces_for_this_image:
            print(f"  No processable faces found in image {i+1} for {person_name}.")

    # --- The rest of the function remains the same ---
    current_idx = 0
    idx_to_name_map = {}

    for name, embeddings_list in temp_embeddings_per_person.items():
        if embeddings_list:
            avg_embedding = np.mean(np.array(embeddings_list), axis=0)
            all_embeddings.append(avg_embedding)
            idx_to_name_map[current_idx] = name
            current_idx += 1
    
    if not all_embeddings:
        print("No embeddings generated. Please check your known faces directory and ensure faces are detectable.")
        return None, None

    embeddings_np = np.array(all_embeddings).astype('float32')
    embedding_dim = embeddings_np.shape[1]

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
# (This function's MTCNN usage is already robust, as it uses mtcnn.detect and converts to numpy)
def recognize_and_draw_faces(image_path, faiss_index, idx_to_name_map, threshold=0.9, output_dir='output_images'):
    """
    Performs face detection, recognition, draws bounding boxes and names,
    and saves the modified image.
    """
    try:
        pil_img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return

    # Convert PIL Image to OpenCV format (NumPy array, BGR channel order)
    img_cv = np.array(pil_img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # Detect faces and get bounding boxes
    # Convert PIL image to NumPy array float32 before passing to mtcnn.detect
    img_np_for_detect = np.array(pil_img).astype(np.float32)
    boxes, probs, landmarks = mtcnn.detect(img_np_for_detect, landmarks=True)

    if boxes is None:
        print(f"No faces detected in {image_path}")
        # Save the original image if no faces were found but still want to save
        output_path = os.path.join(output_dir, os.path.basename(image_path).replace('.', '_no_face.'))
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_path, img_cv)
        print(f"Original image (no faces detected) saved to: {output_path}")
        return

    # Process each detected face
    for i, (box, prob) in enumerate(zip(boxes, probs)):
        x1, y1, x2, y2 = [int(b) for b in box]

        # Crop the face for embedding (using the original PIL image for cleaner crop)
        face_img_pil = pil_img.crop((x1, y1, x2, y2))
        
        # Align the face (MTCNN's post_process does this)
        face_tensor = mtcnn(face_img_pil).unsqueeze(0) # Resize to 160x160 as expected by resnet

        if face_tensor is None:
            print(f"Could not align or resize face {i} in {image_path}. Skipping recognition for this face.")
            continue

        unknown_embedding = resnet(face_tensor.to(device)).detach().cpu().numpy().astype('float32')

        # Search in FAISS index (k=1 for nearest neighbor)
        distances, indices = faiss_index.search(unknown_embedding, 1)
        
        min_distance = distances[0][0]
        closest_index = indices[0][0]

        recognized_name = "Unknown"
        confidence = 0.0

        if min_distance < threshold:
            recognized_name = idx_to_name_map.get(closest_index, "Unknown (Index Error)")
            confidence = 1.0 - (min_distance / threshold)
            if confidence < 0: confidence = 0
        
        display_text = f"{recognized_name} ({confidence:.2f})"
        print(f"Face {i+1}: Detected probability: {prob:.6f}, Min distance: {min_distance:.4f}, Result: {display_text}")

        # --- Drawing on the image ---
        color = (0, 255, 0) # Green color for bounding box (BGR)
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2

        # Draw bounding box
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, thickness)

        # Draw text background for better readability
        (text_width, text_height) = cv2.getTextSize(display_text, font, font_scale, font_thickness)[0]
        # Put text above the bounding box, adjust y for text height
        text_y = y1 - 10 if y1 - 10 > text_height else y2 + text_height + 10 
        text_x = x1

        # Draw a filled rectangle as background for the text
        cv2.rectangle(img_cv, (text_x, text_y - text_height - 5), (text_x + text_width + 5, text_y + 5), (0, 0, 0), -1) # Black background

        # Draw text
        cv2.putText(img_cv, display_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA) # White text

    # --- Save the modified image ---
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.basename(image_path).replace('.', '_recognized.')
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, img_cv)
    print(f"\nProcessed image saved to: {output_path}")

# --- Example Usage ---
known_faces_directory = 'data/known_faces'
faiss_index_file = 'face_recognition_index.faiss'
names_map_file = 'face_names.pkl'
output_images_directory = 'output_images' # Directory to save output images

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

# Load the FAISS index and names if not already loaded
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
        print(f"\n--- Recognizing and drawing on faces in '{unknown_face_image_path}' using FAISS ---")
        recognize_and_draw_faces(
            unknown_face_image_path, faiss_index, idx_to_name_map, threshold=0.9, output_dir=output_images_directory
        )
    else:
        print(f"\nNo image found at '{unknown_face_image_path}'. Please place an image there to test recognition.")