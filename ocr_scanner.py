import cv2
import numpy as np
import pytesseract

# Tesseract executable path setting
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL import Image

def scan_and_extract(image_file):
    """Image ko read karta hai, OpenCV se clean karta hai aur Tesseract se text nikalta hai."""
    try:
        # 1. Image Load
        image = Image.open(image_file)
        img_np = np.array(image)

        # 2. Preprocessing with OpenCV
        # Grayscale me convert karna
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Noise reduce karna (Gaussian Blur)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Text ko dark aur background ko clean white karna (Adaptive Thresholding)
        thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        # 3. OCR Text Extraction
        extracted_text = pytesseract.image_to_string(thresh)

        return extracted_text.strip(), thresh, None
    except Exception as e:
        return None, None, str(e)   
    