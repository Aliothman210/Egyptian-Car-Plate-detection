# main.py
import warnings
warnings.filterwarnings("ignore")

from utils import detect_plate, recognize_easyocr, preprocess_plate, visualize_results


def main():
    # Input image path
    image_path = "test2.jpg"

    # Model path
    model_path = "best.pt"

    # Detect plate
    plates, annotated_img, boxes = detect_plate(model_path, image_path)

    # Preprocess and OCR
    processed_plate = preprocess_plate(plates[0]) if plates else None # You can change the upscale factor if needed (Default is 1.5)

    # OCR text
    if processed_plate is not None:
        text_easy = recognize_easyocr(processed_plate)
        print("EasyOCR result:", text_easy)
    else:
        text_easy = "No plate detected"
        print(text_easy)

    # Visualize
    font_path = "arial.ttf"  # Path to a TTF font file
    visualize_results(annotated_img, processed_plate, text_easy, boxes, font_path)


if __name__ == "__main__":
    main()
