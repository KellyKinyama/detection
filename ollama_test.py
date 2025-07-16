import ollama
import base64
from PIL import Image
import io

def answer_image_question_ollama(image_path, question, model_name='llama3.2-vision'):
    """
    Answers a question about an image using a local Ollama Llama 3.2 Vision model.

    Args:
        image_path (str): Path to the local image file.
        question (str): The question to ask about the image.
        model_name (str): The name of the Llama 3.2 Vision model on Ollama.

    Returns:
        str: The model's answer to the question.
    """
    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()

        print(f"Asking {model_name} about {image_path}: '{question}'")

        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': question,
                    'images': [image_bytes]
                }
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: {e}"

# --- Usage Example ---
if __name__ == "__main__":
    # Ensure you have a relevant image. Let's assume 'traffic_light.jpg' exists.
    # If not, create a dummy image or provide a path to a suitable image.
    # try:
    #     img_q_example = Image.new('RGB', (100, 200), color = 'green')
    #     img_q_example.save('traffic_light.png')
    #     print("Created 'traffic_light.png' for VQA example.")
    # except ImportError:
    #     print("Pillow not installed. Skipping dummy image creation for VQA.")
    
    image_for_qa = 'assets/images.jpg' # Replace with an actual image path
    
    # if image_for_qa == 'assets/images.jpg' and not 'green' in answer_image_question_ollama(image_for_qa, "What color is the light?"):
    #     print("Warning: Dummy image description might not be accurate for VQA. Use a real image.")

    # question1 = "What objects are visible in this image?"
    # answer1 = answer_image_question_ollama(image_for_qa, question1)
    # print(f"\nQuestion: {question1}")
    # print(f"Answer: {answer1}")

    question2 = "Is there any text in this image? If so, what does it say?"
    answer2 = answer_image_question_ollama(image_for_qa, question2)
    print(f"\nQuestion: {question2}")
    print(f"Answer: {answer2}")

    # You would get much better results with a real image containing text or diverse objects.