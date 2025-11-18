# 🚗 Egyptian Car Plate Detection & Recognition System

A complete deep learning solution for detecting and recognizing **Egyptian car license plates** in images. This project combines a custom-trained YOLOv8n model with EasyOCR to automatically detect car plates and extract their text.

## 📋 Table of Contents
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Advanced Configuration](#advanced-configuration)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [OCR & Accuracy](#ocr--accuracy)
- [Preprocessing Options](#preprocessing-options)
- [⚠️ Troubleshooting & Common Problems](#️-troubleshooting--common-problems)
- [Project Files](#project-files)
- [Training Your Own Model](#training-your-own-model)
- [License](#license)

## ✨ Features

- **Custom YOLOv8n Detection**: Trained specifically on Egyptian car plates for high accuracy
- **Real-Time Plate Recognition**: Detects and recognizes plates in images
- **Arabic OCR Support**: Uses EasyOCR with Arabic language support
- **Image Preprocessing**: Automatic grayscale conversion, thresholding, and upscaling
- **Visual Results**: Annotated images with detected plates and recognized text
- **Easy to Use**: Simple command-line interface
- **Adjustable Parameters**: Customize preprocessing and detection settings

## 🏗️ Project Architecture

```
Input Image
    ↓
[YOLO Plate Detection]
    ↓
[Cropped Plate Images]
    ↓
[Image Preprocessing]
    ├─ Grayscale Conversion
    ├─ Otsu Thresholding
    └─ Upscaling
    ↓
[EasyOCR Text Recognition] (Arabic Support)
    ↓
[Visualized Results]
    └─ Annotated image with bounding boxes
    └─ Recognized text overlay
    └─ Processed plate display
```

## 📁 Project Structure

```
Car-Plate-Detection/
│
├── main.py                          # Main entry point script
├── utils.py                         # Utility functions for detection, preprocessing, OCR
├── Plate_detection.ipynb            # Jupyter notebook for model training (Google Colab)
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore file
│
├── best.pt                          # Trained YOLOv8n model
├── yolo11n.pt                       # YOLOv11n base model (optional)
├── yolo8n.pt                        # YOLOv8n base model (optional)
│
├── arial.ttf                        # Font files for text display
├── arial_bold.ttf
├── arial_light.ttf
├── arial_narrow.ttf
│
├── test2.jpg                        # Sample test image
├── output.png                       # Sample output result
├── output2.png                      # Additional sample result
│
├── dataset/                         # (Optional) Local dataset folder
├── EALPR Vehicles dataset/          # Alternative dataset reference
├── Nisan/                           # Sample vehicle images
│
└── README.md                        # This file
```

## 📦 Requirements

- Python 3.8 or higher
- CUDA 11.8+ (Optional, for GPU acceleration)
- 4GB RAM minimum
- Webcam or image files for testing

### Python Libraries:
- ultralytics (YOLOv8)
- easyocr
- opencv-python (cv2)
- numpy
- pillow
- matplotlib
- torch (installed with ultralytics)

## 🚀 Installation

### Step 1: Clone or Download the Project
```bash
git clone <repository-url>
cd Car-Plate-Detection
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install ultralytics easyocr opencv-python numpy pillow matplotlib torch torchvision
```

### Step 4: Verify Installation
```bash
python -c "import torch; print(torch.cuda.is_available())"  # Check GPU availability
python -c "from ultralytics import YOLO; print('YOLO ready')"
```

### Step 5: Download Pre-trained Model

The `best.pt` model should be included in the repository. If not, download it from the release or re-train using the notebook.

---

## 💻 Usage

### Basic Usage

#### 1. Prepare Your Image
Place your car image in the project folder (or note its path):
```
test2.jpg
```

#### 2. Run the Detection Script
```bash
python main.py
```

#### 3. View Results
- Annotated image with detected plate bounding boxes
- Recognized text displayed above the plate
- Processed plate image shown side-by-side

### Example Output:
```
EasyOCR result: "أ ب ع 1456"
```

---

### Advanced Configuration

#### Modify Image Path
Edit `main.py`:
```python
image_path = "path/to/your/image.jpg"  # Change this line
```

#### Adjust Upscale Factor
In `main.py`, modify the preprocessing:
```python
processed_plate = preprocess_plate(plates[0], upscale_factor=2.0)  # Default is 1.5
```

- **Lower values (1.0-1.3)**: Faster processing, less upscaling
- **Default (1.5)**: Balanced performance and accuracy
- **Higher values (2.0+)**: Better OCR accuracy, slower processing

#### Change Font
```python
font_path = "arial_bold.ttf"  # Use different font file
visualize_results(annotated_img, processed_plate, text_easy, boxes, font_path)
```

#### Process Multiple Plates
If your image has multiple plates:
```python
# Current: processes only first plate
processed_plate = preprocess_plate(plates[0])

# Modified: process all plates
for i, plate in enumerate(plates):
    processed_plate = preprocess_plate(plate)
    text_easy = recognize_easyocr(processed_plate)
    print(f"Plate {i+1}: {text_easy}")
```

---

## 📊 Dataset

### Official Dataset
The model was trained on the **Egyptian Cars Plates dataset**:
- **Download**: [https://www.kaggle.com/datasets/mahmoudeldebase/egyptian-cars-plates](https://www.kaggle.com/datasets/mahmoudeldebase/egyptian-cars-plates)
- **Source**: Kaggle
- **Format**: YOLO format with annotations
- **Size**: Thousands of annotated Egyptian car plate images

### Using Your Own Dataset
1. Download from the Kaggle link above
2. Extract to a local folder
3. Structure should be YOLO format:
   ```
   dataset/
   ├── images/
   │   ├── train/
   │   ├── val/
   │   └── test/
   └── labels/
       ├── train/
       ├── val/
       └── test/
   ```

### Alternative Datasets
- **EALPR (Egyptian Automatic License Plate Recognition)**
- **General car plate datasets** (may require fine-tuning)

---

## 🤖 Model Details

### Model Architecture
- **Base Model**: YOLOv8 (Nano variant)
- **Variant**: `yolov8n.pt`
- **Training**: Custom trained on Egyptian car plate dataset using Google Colab
- **Inference Speed**: ~30-50ms per image (CPU), ~10-20ms (GPU)
- **Model Size**: ~6.3 MB

### Training Specifications
- **Framework**: Ultralytics YOLOv8
- **Dataset**: Egyptian Cars Plates (Kaggle)
- **Training Environment**: Google Colab with GPU
- **Epochs**: Configured in training notebook
- **Input Size**: 640x640
- **Batch Size**: Configured during training

### Model Performance
- Optimized for Egyptian license plates
- Handles various lighting conditions
- Works with different vehicle types
- Accurate bounding box detection

---

## 🔤 OCR & Accuracy

### Text Recognition Technology
- **Library**: EasyOCR
- **Language**: Arabic (ar)
- **Model**: Pre-trained OCR model for Arabic text

### Important Notes on Accuracy

⚠️ **OCR accuracy may not be 100%** due to:

1. **Single Image Dependency**: Recognition is based on a single photo only
   - Different angles, lighting, and distances affect accuracy
   - Blurry or partially visible plates are harder to recognize

2. **Arabic Font Variations**: Car plates may use different Arabic fonts
   - Some characters might be misrecognized
   - Similar-looking characters can be confused

3. **Why We Don't Train Custom OCR**:
   - Training a separate OCR model for each Arabic letter would be extremely time-consuming
   - Would require thousands of annotated samples per character
   - Diminishing returns on accuracy improvement
   - Pre-trained EasyOCR provides good general-purpose Arabic recognition

### Improving Recognition Accuracy

**Best Practices**:
- ✅ Use clear, well-lit images
- ✅ Ensure plate is facing the camera
- ✅ Avoid extreme angles or shadows
- ✅ Increase upscale factor (2.0-3.0) for small/distant plates
- ✅ Use high-resolution input images (1920x1080+)

**Workarounds for Poor Recognition**:
```python
# Increase preprocessing quality
processed_plate = preprocess_plate(plates[0], upscale_factor=2.5)

# Or adjust detection confidence if needed in detect_plate()
```

---

## 🔧 Preprocessing Options

The preprocessing pipeline improves OCR accuracy through several steps:

### 1. Grayscale Conversion
Converts BGR color image to grayscale for better contrast:
```python
gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
```

### 2. Otsu Thresholding
Automatic threshold selection for binary image:
```python
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```
- Maximizes contrast between text and background
- Creates cleaner image for OCR

### 3. Image Upscaling
Enlarges image for better character recognition:
```python
thresh_up = cv2.resize(thresh, (upscale_w, upscale_h), interpolation=cv2.INTER_CUBIC)
```
- Default: 1.5x upscale
- Cubic interpolation preserves quality

### Customizing Preprocessing

To modify preprocessing, edit `preprocess_plate()` in `utils.py`:

```python
def preprocess_plate(plate_img, upscale_factor=1.5):
    # ... existing code ...
    
    # Optional: Add additional preprocessing
    # denoising, morphological operations, etc.
```

---

## ⚠️ Troubleshooting & Common Problems

### Problem 1: "No module named 'ultralytics'" or "No module named 'easyocr'"

**Error Message:**
```
ModuleNotFoundError: No module named 'ultralytics'
ModuleNotFoundError: No module named 'easyocr'
```

**Solution:**
```bash
# Reinstall all requirements
pip install -r requirements.txt

# Or install individually
pip install ultralytics easyocr
```

---

### Problem 2: YOLO Model Download Failure

**Error Messages:**
- `Model could not be found`
- `URLError: urlopen error`
- `Connection timeout`

**Solution:**

The model should already be in your repository as `best.pt`. If you need to download base models:

1. **Download YOLOv8n Model**:
   ```
   https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt
   ```

2. **Download YOLOv11n Model** (Optional):
   ```
   https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11n.pt
   ```

3. **Save to project folder** and update `main.py`:
   ```python
   model_path = "yolov8n.pt"  # or "best.pt"
   ```

**Automatic Download (First Run)**:
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # Downloads automatically if not found
```

---

### Problem 3: EasyOCR Model Download Issues

**Error Messages:**
- `Download failed`
- `Connection error`
- `urllib3.exceptions.MaxRetryError`

**Solution:**

EasyOCR models are downloaded on first use. If this fails:

1. **Manual Download**:
   - Download from: https://github.com/JaidedAI/EasyOCR/releases
   - Place in: `~/.EasyOCR/model/`

2. **Or try alternative approach**:
   ```bash
   # Clear cache and reinstall
   pip uninstall easyocr
   pip install easyocr
   ```

3. **Use CPU if GPU issues**:
   ```python
   # In utils.py, change:
   reader = easyocr.Reader(['ar'], gpu=False)  # Force CPU
   ```

---

### Problem 4: "No Plate Detected"

**Error Message:**
```
text_easy = "No plate detected"
```

**Causes & Solutions**:

1. **Image Quality Issues**:
   - ✅ Use clearer, well-lit images
   - ✅ Ensure plate is visible and facing camera
   - ✅ Avoid extreme angles

2. **Model Confidence Too High**:
   - Lower detection confidence threshold in `detect_plate()`:
   ```python
   results = model.predict(image_path, verbose=False, conf=0.3)
   ```

3. **Wrong Model**:
   - Ensure you're using `best.pt` (trained on Egyptian plates)
   - Not a generic YOLO model

4. **Image Format Issues**:
   - Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
   - Ensure image is not corrupted

---

### Problem 5: Poor OCR Accuracy / Wrong Text Recognition

**Symptoms**:
- Numbers/letters misrecognized
- Partial text recognition
- Special characters not recognized

**Solutions**:

1. **Increase Image Upscaling**:
   ```python
   processed_plate = preprocess_plate(plates[0], upscale_factor=2.5)
   ```

2. **Improve Image Quality**:
   - Use better quality input images
   - Ensure proper lighting
   - Reduce motion blur

3. **Check Plate Visibility**:
   - Plate should be straight and clear
   - Text should be readable by human eye
   - No extreme shadows or glare

4. **Note**: With EasyOCR, some variations in recognition are normal
   - This is due to Arabic character variations on plates
   - Training a custom OCR would be impractical

---

### Problem 6: CUDA/GPU Not Available

**Error Message**:
```
UserWarning: CUDA device not available
CUDA is not available, using CPU instead
```

**Info**: This is a warning, not an error. The system will still work on CPU.

**To Use GPU** (if available):

1. **Check GPU availability**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Install GPU support**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Verify CUDA**:
   ```bash
   nvidia-smi  # Check NVIDIA GPU
   ```

---

### Problem 7: Font Not Found Error

**Error Message**:
```
Font file 'arial.ttf' not found
```

**Solution**:

1. **Use provided fonts**:
   - Use one of the included font files:
   ```python
   font_path = "arial_bold.ttf"
   # or
   font_path = "arial_light.ttf"
   ```

2. **System font fallback**:
   - If no font found, system default is used (less pretty but functional)

3. **Add custom font**:
   - Add your `.ttf` file to project folder
   - Update `main.py`:
   ```python
   font_path = "your_font.ttf"
   ```

---

### Problem 8: Memory Issues / Out of Memory

**Error Message**:
```
RuntimeError: CUDA out of memory
MemoryError: Unable to allocate X MB
```

**Solutions**:

1. **Use CPU instead of GPU**:
   ```python
   # In utils.py
   reader = easyocr.Reader(['ar'], gpu=False)
   ```

2. **Reduce image size**:
   ```python
   # Resize before detection
   img = cv2.imread(image_path)
   img = cv2.resize(img, (640, 480))
   ```

3. **Lower batch size** (if processing multiple images)

---

### Problem 9: Slow Processing Speed

**Symptoms**:
- Takes >30 seconds per image
- High CPU usage

**Solutions**:

1. **Use GPU acceleration** (see Problem 6)

2. **Reduce upscale factor**:
   ```python
   processed_plate = preprocess_plate(plates[0], upscale_factor=1.0)
   ```

3. **Use smaller input image**:
   ```python
   img = cv2.imread(image_path)
   img = cv2.resize(img, (800, 600))
   ```

4. **Disable verbose output**:
   ```python
   results = model.predict(image_path, verbose=False)
   ```

---

### Problem 10: Matplotlib Display Issues (Linux/WSL)

**Error Message**:
```
RuntimeError: main thread is not in main loop
```

**Solution**:

Add to the beginning of `main.py`:
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' for non-interactive
```

---

## 📂 Project Files

### Core Files
- **main.py**: Entry point, orchestrates the detection pipeline
- **utils.py**: Contains all detection, preprocessing, and visualization functions

### Model Files
- **best.pt**: Trained YOLOv8n model on Egyptian car plates
- **yolo8n.pt**: Base YOLOv8 Nano model (optional backup)
- **yolo11n.pt**: Base YOLOv11 Nano model (optional alternative)

### Font Files
- **arial.ttf**: Standard Arial font
- **arial_bold.ttf**: Bold Arial font
- **arial_light.ttf**: Light Arial font
- **arial_narrow.ttf**: Narrow Arial font

### Configuration & Documentation
- **requirements.txt**: Python dependencies
- **Plate_detection.ipynb**: Training notebook (Google Colab)
- **data.yaml**: YOLO dataset configuration
- **.gitignore**: Git ignore rules

### Sample Files
- **test2.jpg**: Sample test image
- **output.png**: Example detection result
- **output2.png**: Additional example result

---

## 🏋️ Training Your Own Model

For detailed training instructions, refer to **Plate_detection.ipynb**:

### Quick Overview:
1. Prepare dataset in YOLO format
2. Use Google Colab for GPU acceleration
3. Train using Ultralytics YOLOv8
4. Export best model as `best.pt`

### Download Training Dataset:
```
https://www.kaggle.com/datasets/mahmoudeldebase/egyptian-cars-plates
```

### Basic Training Code:
```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    device=0  # GPU device ID
)

# Save best model
model.save('best.pt')
```

---

## 📝 License

[Add your license information here - e.g., MIT, GPL, etc.]

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests
- Add support for additional plate types

---

## 📧 Support & Resources

### Official Documentation
- **Ultralytics YOLOv8**: https://github.com/ultralytics/ultralytics
- **EasyOCR**: https://github.com/JaidedAI/EasyOCR
- **OpenCV**: https://docs.opencv.org/

### Dataset
- **Egyptian Cars Plates Dataset**: https://www.kaggle.com/datasets/mahmoudeldebase/egyptian-cars-plates

### Related Projects
- YOLOv8 License Plate Detection
- Automatic Number Plate Recognition (ANPR)
- Arabic OCR Systems

---

## 🎯 Future Improvements

Potential enhancements:
- ✅ Support for multiple plates per image (full pipeline)
- ✅ Real-time video stream processing
- ✅ Batch image processing
- ✅ Database storage of recognized plates
- ✅ Confidence score filtering
- ✅ Multi-language OCR support
- ✅ Web API endpoint
- ✅ Mobile app integration
- ✅ Plate tracking across frames
- ✅ Statistical analysis of detections

---

## 🎓 Project Credits

**Dataset Source**: Mahmoud Eldebase - Egyptian Cars Plates (Kaggle)
**Technologies**: 
- Ultralytics YOLOv8
- EasyOCR
- OpenCV
- Python

---

**Last Updated**: 2025
**Version**: 1.0

**Status**: Active & Maintained
