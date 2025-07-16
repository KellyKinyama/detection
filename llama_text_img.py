from pydantic import BaseModel
import requests
import json
import uvicorn
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Form # Import Form for text fields with file uploads
import base64 # Import base64 for encoding images

app = FastAPI()

# This Pydantic model is for the text-only endpoint
class Prompt(BaseModel):
    prompt: str

@app.post("/generate_text_only") # Renamed for clarity - original functionality
def generate_text_only(prompt: Prompt):
    """
    Generates text based on a text-only prompt using Ollama's /api/generate endpoint.
    """
    try:
        # Use environment variables for host and model, with fallbacks
        # Changed to string literals as in your request, but environment variables are good practice
        ollama_host = "http://localhost:11434"
        ollama_model = "llama3.2-vision"

        response = requests.post(
            f"{ollama_host}/api/generate",
            json={"model": ollama_model, "prompt": prompt.prompt, "stream": True}, # Added stream explicitly
            stream=True,
            timeout=120
        )
        response.raise_for_status()

        output = ""
        for line in response.iter_lines():
            if line:
                data = line.decode("utf-8").strip()
                if data.startswith("data: "):
                    data = data[len("data: "):]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    output += chunk.get("response") or chunk.get("text") or ""
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from line: {data}")
                    continue

        return {"response": output.strip() or "(Empty response from model)"}

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Ollama request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.post("/generate_with_image")
async def generate_with_image(
    # FastAPI can directly handle file uploads using UploadFile
    image: UploadFile = File(...), # The uploaded image file
    # Use Form for text fields when also uploading a file (multipart/form-data)
    prompt: str = Form("Describe this image in detail.") # The text prompt/question
):
    """
    Generates a response to a prompt about an uploaded image using Ollama's /api/generate endpoint.
    """
    try:
        ollama_host = "http://localhost:11434"
        ollama_model = "llama3.2-vision"

        # Read the image content as bytes
        image_bytes = await image.read()

        # Base64 encode the image bytes
        # Ollama's API expects base64 encoded image strings in the 'images' array
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        # Prepare the payload for Ollama's /api/generate endpoint
        # For multimodal models, the 'images' field is used
        ollama_payload = {
            "model": ollama_model,
            "prompt": prompt,
            "images": [encoded_image], # List containing the base64 encoded image
            "stream": True # Keep streaming for potentially long responses
        }

        response = requests.post(
            f"{ollama_host}/api/generate",
            json=ollama_payload,
            stream=True,
            timeout=300 # Increased timeout for image processing, as it can take longer
        )
        response.raise_for_status()

        output = ""
        for line in response.iter_lines():
            if line:
                data = line.decode("utf-8").strip()
                if data.startswith("data: "):
                    data = data[len("data: "):]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    # For /api/generate, the response is usually directly in 'response' or 'text'
                    output += chunk.get("response") or chunk.get("text") or ""
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from line: {data}")
                    continue

        return {"response": output.strip() or "(Empty response from model)"}

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Ollama request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False) # reload=True for development