# from fastapi import FastAPI, UploadFile, File, HTTPException
# from PIL import Image
# import requests
# from io import BytesIO
# import base64

# import torch
# from torch import nn
# from torchvision.models import resnet50
# import torchvision.transforms as T
# torch.set_grad_enabled(False)

# # --- DETR Model Definition (your existing code) ---
# class DETRdemo(nn.Module):
#     """
#     Demo DETR implementation.
#     ... (rest of your DETRdemo class definition)
#     """
#     def __init__(self, num_classes, hidden_dim=256, nheads=8,
#                  num_encoder_layers=6, num_decoder_layers=6):
#         super().__init__()

#         # create ResNet-50 backbone
#         self.backbone = resnet50()
#         del self.backbone.fc

#         # create conversion layer
#         self.conv = nn.Conv2d(2048, hidden_dim, 1)

#         # create a default PyTorch transformer
#         self.transformer = nn.Transformer(
#             hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

#         # prediction heads, one extra class for predicting non-empty slots
#         # note that in baseline DETR linear_bbox layer is 3-layer MLP
#         self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
#         self.linear_bbox = nn.Linear(hidden_dim, 4)

#         # output positional encodings (object queries)
#         self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

#         # spatial positional encodings
#         # note that in baseline DETR we use sine positional encodings
#         self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
#         self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

#     def forward(self, inputs):
#         # propagate inputs through ResNet-50 up to avg-pool layer
#         x = self.backbone.conv1(inputs)
#         x = self.backbone.bn1(x)
#         x = self.backbone.relu(x)
#         x = self.backbone.maxpool(x)

#         x = self.backbone.layer1(x)
#         x = self.backbone.layer2(x)
#         x = self.backbone.layer3(x)
#         x = self.backbone.layer4(x)

#         # convert from 2048 to 256 feature planes for the transformer
#         h = self.conv(x)

#         # construct positional encodings
#         H, W = h.shape[-2:]
#         pos = torch.cat([
#             self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
#             self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
#         ], dim=-1).flatten(0, 1).unsqueeze(1)

#         # propagate through the transformer
#         h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
#                              self.query_pos.unsqueeze(1)).transpose(0, 1)

#         # finally project transformer outputs to class labels and bounding boxes
#         return {'pred_logits': self.linear_class(h),
#                 'pred_boxes': self.linear_bbox(h).sigmoid()}

# # --- Global Model and Utility Initialization ---
# # This part runs once when the API starts
# detr = DETRdemo(num_classes=91)
# state_dict = torch.hub.load_state_dict_from_url(
#     url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
#     map_location='cpu', check_hash=True)
# detr.load_state_dict(state_dict)
# detr.eval() # Set model to evaluation mode

# # COCO classes
# CLASSES = [
#     'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#     'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
#     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
#     'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
#     'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
#     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
#     'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
#     'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
#     'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
#     'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
#     'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
#     'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
#     'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
#     'toothbrush'
# ]

# # standard PyTorch mean-std input image normalization
# transform = T.Compose([
#     T.Resize(800),
#     T.ToTensor(),
#     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# # for output bounding box post-processing
# def box_cxcywh_to_xyxy(x):
#     x_c, y_c, w, h = x.unbind(1)
#     b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
#          (x_c + 0.5 * w), (y_c + 0.5 * h)]
#     return torch.stack(b, dim=1)

# def rescale_bboxes(out_bbox, size):
#     img_w, img_h = size
#     b = box_cxcywh_to_xyxy(out_bbox)
#     b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
#     return b

# def detect_objects(im: Image.Image, model, transform):
#     # mean-std normalize the input image (batch-size: 1)
#     img = transform(im).unsqueeze(0)

#     # demo model only support by default images with aspect ratio between 0.5 and 2
#     # if you want to use images with an aspect ratio outside this range
#     # rescale your image so that the maximum size is at most 1333 for best results
#     if not (img.shape[-2] <= 1600 and img.shape[-1] <= 1600):
#         raise HTTPException(status_code=400, detail='Image too large. Demo model only supports images up to 1600 pixels on each side.')

#     # propagate through the model
#     outputs = model(img)

#     # keep only predictions with 0.7+ confidence
#     probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
#     keep = probas.max(-1).values > 0.7

#     # convert boxes from [0; 1] to image scales
#     bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

#     results = []
#     for p, (xmin, ymin, xmax, ymax) in zip(probas[keep], bboxes_scaled.tolist()):
#         cl = p.argmax().item()
#         label = CLASSES[cl]
#         confidence = p[cl].item()
#         results.append({
#             "label": label,
#             "confidence": round(confidence, 4),
#             "box": {
#                 "xmin": round(xmin, 2),
#                 "ymin": round(ymin, 2),
#                 "xmax": round(xmax, 2),
#                 "ymax": round(ymax, 2)
#             }
#         })
#     return results

# # --- FastAPI Application ---
# app = FastAPI(
#     title="DETR Object Detection API",
#     description="An API for performing object detection using a pre-trained DETR model.",
#     version="1.0.0",
# )

# @app.get("/")
# async def root():
#     return {"message": "Welcome to the DETR Object Detection API! Use /detect_from_url or /detect_from_upload."}

# @app.post("/detect_from_url")
# async def detect_from_url(image_url: str):
#     """
#     Performs object detection on an image from a given URL.
#     """
#     try:
#         response = requests.get(image_url, stream=True)
#         response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
#         im = Image.open(BytesIO(response.content)).convert('RGB')
#     except requests.exceptions.RequestException as e:
#         raise HTTPException(status_code=400, detail=f"Could not fetch image from URL: {e}")
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Could not process image from URL: {e}")

#     try:
#         detections = detect_objects(im, detr, transform)
#         return {"filename": image_url.split('/')[-1], "detections": detections}
#     except HTTPException as e:
#         raise e # Re-raise if it's already an HTTPException
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Detection failed: {e}")

# @app.post("/detect_from_upload")
# async def detect_from_upload(file: UploadFile = File(...)):
#     """
#     Performs object detection on an uploaded image file.
#     """
#     if not file.content_type.startswith("image/"):
#         raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

#     try:
#         contents = await file.read()
#         im = Image.open(BytesIO(contents)).convert('RGB')
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Could not read or open image file: {e}")

#     try:
#         detections = detect_objects(im, detr, transform)
#         return {"filename": file.filename, "detections": detections}
#     except HTTPException as e:
#         raise e # Re-raise if it's already an HTTPException
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Detection failed: {e}")

# # Optional: Endpoint to return the image with bounding boxes
# # This requires matplotlib, and is usually not recommended for production APIs
# # as it increases payload size and dependencies.
# # For demonstration purposes:
# # @app.post("/detect_and_draw_boxes")
# # async def detect_and_draw_boxes(file: UploadFile = File(...)):
# #     """
# #     Performs object detection on an uploaded image and returns the image
# #     with bounding boxes drawn on it (base64 encoded).
# #     """
# #     if not file.content_type.startswith("image/"):
# #         raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

# #     try:
# #         contents = await file.read()
# #         im = Image.open(BytesIO(contents)).convert('RGB')
# #     except Exception as e:
# #         raise HTTPException(status_code=400, detail=f"Could not read or open image file: {e}")

# #     try:
# #         probas, bboxes_scaled = detect_objects(im, detr, transform)

# #         plt.figure(figsize=(16,10))
# #         plt.imshow(im)
# #         ax = plt.gca()
# #         for p, (xmin, ymin, xmax, ymax), c in zip(probas, bboxes_scaled.tolist(), COLORS * 100):
# #             ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
# #                                         fill=False, color=c, linewidth=3))
# #             cl = p.argmax()
# #             text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
# #             ax.text(xmin, ymin, text, fontsize=15,
# #                     bbox=dict(facecolor='yellow', alpha=0.5))
# #         plt.axis('off')

# #         buffer = BytesIO()
# #         plt.savefig(buffer, format='PNG', bbox_inches='tight', pad_inches=0)
# #         plt.close() # Close the plot to free memory
# #         encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
# #         return {"filename": file.filename, "image_with_boxes": encoded_image, "mime_type": "image/png"}

# #     except HTTPException as e:
# #         raise e
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"Detection failed: {e}")

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import base64

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False)

# --- DETR Model Definition (your existing code) ---
class DETRdemo(nn.Module):
    """
    Demo DETR implementation.
    ... (rest of your DETRdemo class definition)
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)

        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h),
                'pred_boxes': self.linear_bbox(h).sigmoid()}

# --- Global Model and Utility Initialization ---
# This part runs once when the API starts
detr = DETRdemo(num_classes=91)
state_dict = torch.hub.load_state_dict_from_url(
    url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
    map_location='cpu', check_hash=True)
detr.load_state_dict(state_dict)
detr.eval() # Set model to evaluation mode

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization (you can customize these)
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [.494, .184, .556], [.466, .674, .188], [.301, .745, .933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def detect_objects(im: Image.Image, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    if not (img.shape[-2] <= 1600 and img.shape[-1] <= 1600):
        raise HTTPException(status_code=400, detail='Image too large. Demo model only supports images up to 1600 pixels on each side.')

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    results = []
    for p, (xmin, ymin, xmax, ymax) in zip(probas, bboxes_scaled.tolist()):
        cl = p.argmax().item()
        label = CLASSES.get(cl, 'UNKNOWN') # Handle potential out-of-bounds indices
        confidence = p.max().item()
        results.append({
            "label": label,
            "confidence": round(confidence, 4),
            "box": {
                "xmin": round(xmin, 2),
                "ymin": round(ymin, 2),
                "xmax": round(xmax, 2),
                "ymax": round(ymax, 2)
            }
        })
    return probas, bboxes_scaled, im.size # Return probas, boxes, and image size

def draw_bounding_boxes(image: Image.Image, probas, boxes):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    try:
        font = ImageFont.truetype("arial.ttf", size=int(height / 25)) # Adjust font size as needed
    except IOError:
        font = ImageFont.load_default()

    for i, (p, (xmin, ymin, xmax, ymax)) in enumerate(zip(probas, boxes.tolist())):
        cl = p.argmax().item()
        label = CLASSES.get(cl, 'UNKNOWN')
        confidence = f"{p.max().item():0.2f}"
        text = f"{label}: {confidence}"
        color = COLORS[(cl * 3) % len(COLORS)] # Cycle through colors, multiply by 3 for more variation
        color_rgb = tuple(int(c * 255) for c in color)

        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color_rgb, width=3)
        text_width, text_height = draw.textsize(text, font=font)
        draw.rectangle([(xmin, ymin - text_height - 5), (xmin + text_width + 5, ymin)], fill=color_rgb)
        draw.text((xmin + 2, ymin - text_height - 5), text, fill=(255, 255, 255), font=font)
    return image

# --- FastAPI Application ---
app = FastAPI(
    title="DETR Object Detection API",
    description="An API for performing object detection using a pre-trained DETR model.",
    version="1.0.0",
)

@app.get("/")
async def root():
    return {"message": "Welcome to the DETR Object Detection API! Use /detect_from_url, /detect_from_upload, or /detect_and_draw."}

@app.post("/detect_from_url")
async def detect_from_url(image_url: str):
    """
    Performs object detection on an image from a given URL and returns JSON results.
    """
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        im = Image.open(BytesIO(response.content)).convert('RGB')
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch image from URL: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image from URL: {e}")

    try:
        probas, bboxes_scaled, _ = detect_objects(im, detr, transform)
        results = []
        for p, (xmin, ymin, xmax, ymax) in zip(probas, bboxes_scaled.tolist()):
            cl = p.argmax().item()
            label = CLASSES.get(cl, 'UNKNOWN')
            confidence = p.max().item()
            results.append({
                "label": label,
                "confidence": round(confidence, 4),
                "box": {
                    "xmin": round(xmin, 2),
                    "ymin": round(ymin, 2),
                    "xmax": round(xmax, 2),
                    "ymax": round(ymax, 2)
                }
            })
        return {"filename": image_url.split('/')[-1], "detections": results}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")

@app.post("/detect_from_upload")
async def detect_from_upload(file: UploadFile = File(...)):
    """
    Performs object detection on an uploaded image file and returns JSON results.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        contents = await file.read()
        im = Image.open(BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read or open image file: {e}")

    try:
        probas, bboxes_scaled, _ = detect_objects(im, detr, transform)
        results = []
        for p, (xmin, ymin, xmax, ymax) in zip(probas, bboxes_scaled.tolist()):
            cl = p.argmax().item()
            label = CLASSES.get(cl, 'UNKNOWN')
            confidence = p.max().item()
            results.append({
                "label": label,
                "confidence": round(confidence, 4),
                "box": {
                    "xmin": round(xmin, 2),
                    "ymin": round(ymin, 2),
                    "xmax": round(xmax, 2),
                    "ymax": round(ymax, 2)
                }
            })
        return {"filename": file.filename, "detections": results}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")

@app.post("/detect_and_draw")
async def detect_and_draw(file: UploadFile = File(...)):
    """
    Performs object detection on an uploaded image and returns the image
    with bounding boxes drawn on it (base64 encoded).
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        contents = await file.read()
        im = Image.open(BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read or open image file: {e}")

    try:
        probas, bboxes_scaled, _ = detect_objects(im, detr, transform)
        image_with_boxes = draw_bounding_boxes(im.copy(), probas, bboxes_scaled) # Use a copy to avoid modifying the original

        buffer = BytesIO()
        image_with_boxes.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "filename": file.filename,
            "image_base64": encoded_image,
            "mime_type": "image/png",
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection and drawing failed: {e}")

@app.post("/detect_and_draw_from_url")
async def detect_and_draw_from_url(image_url: str):
    """
    Performs object detection on an image from a URL and returns the image
    with bounding boxes drawn on it (base64 encoded).
    """
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        im = Image.open(BytesIO(response.content)).convert('RGB')
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch image from URL: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image from URL: {e}")

    try:
        probas, bboxes_scaled, _ = detect_objects(im, detr, transform)
        image_with_boxes = draw_bounding_boxes(im.copy(), probas, bboxes_scaled)

        buffer = BytesIO()
        image_with_boxes.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "filename": image_url.split('/')[-1],
            "image_base64": encoded_image,
            "mime_type": "image/png",
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection and drawing failed: {e}")