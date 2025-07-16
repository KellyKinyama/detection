import ollama
import base64
from PIL import Image
import io

# ... (function definitions from previous examples) ...

def get_image_description_ollama(image_path, model_name='llama3.2-vision'):
    try:
        # Load image and convert to bytes
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read() # <-- THIS IS THE CRITICAL PART

        print(f"Asking {model_name} to describe the image: {image_path}...")

        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': 'Describe this image in detail. What are the main objects, activities, and colors?',
                    'images': [image_bytes] # <-- PASSING RAW BYTES
                }
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: {e}"

def answer_image_question_ollama(image_path, question, model_name='llama3.2-vision'):
    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read() # <-- THIS IS THE CRITICAL PART

        print(f"Asking {model_name} about {image_path}: '{question}'")

        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': question,
                    'images': [image_bytes] # <-- PASSING RAW BYTES
                }
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: {e}"

# --- Your main execution block ---
if __name__ == "__main__":
    image_file = r'C:\www\python\detection\assets\images.jpg' # Use full path for certainty

    # Create dummy image if it doesn't exist for the VQA test message
    # This part of the code is only for creating the dummy.
    # Your 'assets/images.jpg' is likely a real image.
    # if not os.path.exists('traffic_light.png'):
    #     try:
    #         img_q_example = Image.new('RGB', (100, 200), color='green')
    #         img_q_example.save('traffic_light.png')
    #         print("Created 'traffic_light.png' for VQA example.")
    #     except ImportError:
    #         print("Pillow not installed or issue creating dummy image.")


    question1 = "What objects are visible in this image?"
    answer1 = answer_image_question_ollama(image_file, question1)
    print(f"\nQuestion: {question1}")
    print(f"Answer: {answer1}")

    question2 = "Is there any text in this image? If so, what does it say?"
    answer2 = answer_image_question_ollama(image_file, question2)
    print(f"\nQuestion: {question2}")
    print(f"Answer: {answer2}")

    # You can add the image captioning part back if you want to test it too
    # description = get_image_description_ollama(image_file)
    # print("\nGenerated Description:")
    # print(description)