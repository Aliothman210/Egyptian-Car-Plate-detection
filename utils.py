# utils.py
import cv2
import os
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import easyocr
import matplotlib.pyplot as plt

# 1) detect_plate function

def detect_plate(model_path, image_path):
    """Detect plates with YOLO. Return (cropped_plates, annotated_img, boxes)."""
    model = YOLO(model_path)
    results = model.predict(image_path, verbose=False)
    # Read image
    img = cv2.imread(image_path)
    annotated = img.copy()

    cropped_plates = [] # list of cropped plate images
    boxes = [] # list of (x1,y1,x2,y2) tuples for visualization purposes with visualize_results

    # Extract boxes and crops
    if len(results) > 0 and hasattr(results[0], "boxes"):
        xyxy = results[0].boxes.xyxy.cpu().numpy().astype(int)
        for box in xyxy:
            x1, y1, x2, y2 = map(int, box)
            boxes.append((x1, y1, x2, y2)) # for visualize_results
            # Crop plate from original image
            crop = img[y1:y2, x1:x2]
            cropped_plates.append(crop)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return cropped_plates, annotated, boxes


# 2) preprocess_plate function
def preprocess_plate(plate_img, upscale_factor=1.5):
    """
    Apply preprocessing steps to improve OCR accuracy:
    - Grayscale
    - Thresholding (Otsu)
    - Upscale
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Thresholding
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Upscale
    h, w = thresh.shape[:2]
    upscale_w, upscale_h = int(w * upscale_factor), int(h * upscale_factor)
    thresh_up = cv2.resize(thresh, (upscale_w, upscale_h), interpolation=cv2.INTER_CUBIC)

    return thresh_up


# 3) recognize_easyocr: tries GPU then CPU fallback
def recognize_easyocr(plate_img):
    """
    Recognize plate text using EasyOCR.
    """
    os.environ["FLAGS_use_mkldnn"] = "0"
    reader = easyocr.Reader(['ar'], gpu=True)
    results = reader.readtext(plate_img)

    texts = [res[1] for res in results]
    return " | ".join(texts)


# 4) visualize_results function
def visualize_results(annotated_img, plate_img, text, boxes=None, font_path="arial.ttf"):
    """
    Display annotated image with OCR text drawn above the first box, and show processed plate.
    annotated_img: BGR image with YOLO rectangles already drawn
    plate_img: BGR processed crop (or None)
    text: OCR text (string)
    boxes: list of (x1,y1,x2,y2) from detect_plate (optional)
    """
    # convert to RGB PIL image
    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(annotated_rgb).convert("RGB")
    draw = ImageDraw.Draw(pil_img)

    # choose font size based on box height if available
    font_size = 28
    if boxes and len(boxes) > 0:
        x1, y1, x2, y2 = boxes[0]
        box_h = max(12, y2 - y1)
        font_size = max(16, int(box_h * 0.45))

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    # position text above first box if available, else top-left
    if boxes and len(boxes) > 0:
        x1, y1, x2, y2 = boxes[0]
        # measure text size
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = draw.textsize(text, font=font)

        tx = x1
        ty = max(0, y1 - text_h - 6)
    else:
        # fallback position
        tx, ty = 10, 10
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = draw.textsize(text, font=font)

    # draw solid background rectangle for readability
    rect_coords = [(tx - 3, ty - 3), (tx + text_w + 3, ty + text_h + 3)]
    draw.rectangle(rect_coords, fill=(0, 0, 0))
    draw.text((tx, ty), text, font=font, fill=(255, 255, 255))

    # prepare matplotlib display
    cols = 2 if plate_img is not None else 1
    fig, axes = plt.subplots(1, cols, figsize=(12, 6))
    if cols == 2:
        ax0, ax1 = axes
    else:
        ax0 = axes

    ax0.imshow(pil_img)
    ax0.set_title("Detected Plate with OCR Text")
    ax0.axis("off")

    if plate_img is not None:
        # plate_img is BGR; convert to RGB
        plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
        ax1.imshow(plate_rgb)
        ax1.set_title("Processed Plate")
        ax1.axis("off")

    plt.tight_layout()
    plt.show()

    return pil_img
