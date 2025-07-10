import google.generativeai as genai
from PIL import Image

# Configure your API key (replace with your actual key or load from environment variables)
# genai.configure(api_key="YOUR_GEMINI_API_KEY")

# Create a GenerativeModel instance
# model = genai.GenerativeModel('gemini-pro-vision') # For single image
model = genai.GenerativeModel('gemini-1.5-flash') # Recommended for faster processing and longer context

def extract_text_with_gemini(image_path, prompt="Extract all text from this image."):
    """
    Extracts text from an image using a Google Gemini model.
    """
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        return f"Error: Image not found at {image_path}"
    except Exception as e:
        return f"Error loading image {image_path}: {e}"

    # Prepare the content for the model
    # The prompt can be simple or very specific, e.g., "What is the license plate number?"
    # or "Extract all text in a JSON format with 'text_blocks' and their 'bounding_boxes'."
    content = [
        prompt,
        img
    ]

    try:
        response = model.generate_content(content)
        return response.text
    except Exception as e:
        return f"Error during API call: {e}"

# --- Example Usage ---
image_file = 'path/to/your/image_with_text.jpg' # Replace with your image path

# Example 1: General text extraction
extracted_text = extract_text_with_gemini(image_file, "Extract all text you can read in this image.")
print("--- General Text Extraction ---")
print(extracted_text)

# Example 2: More specific extraction (e.g., for a license plate)
extracted_plate_number = extract_text_with_gemini(image_file, "What is the license plate number visible in this image? Only provide the number.")
print("\n--- License Plate Number Extraction ---")
print(extracted_plate_number)

# Example 3: Extracting structured data (e.g., from an invoice)
# This requires a more complex prompt and potentially fine-tuning
invoice_text = extract_text_with_gemini(image_file, "Extract the invoice number, total amount, and date from this invoice. Return in JSON format: {'invoice_number': '', 'total_amount': '', 'date': ''}")
print("\n--- Structured Data Extraction ---")
print(invoice_text)