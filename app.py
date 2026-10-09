import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re

# Page Configuration
st.set_page_config(page_title="AI Document Scanner", layout="wide")

st.title("📚 AI Document Scanner & Image OCR Engine")
st.markdown("Scan documents with real-time shadow removal and manual cropping, or extract text from uploaded images.")

# Navigation Tabs
tab1, tab2 = st.tabs([
    "📷 Document Scanner", 
    "🖼️ Image Uploader"
])

def clean_ocr_text(raw_text):
    """Filters out noise and garbage characters from OCR output."""
    lines = raw_text.split("\n")
    cleaned_lines = []
    for line in lines:
        # Remove lines that contain mostly garbage or single random symbols
        line_str = line.strip()
        if len(line_str) > 2 or any(c.isalnum() for c in line_str):
            cleaned_lines.append(line_str)
    return "\n".join(cleaned_lines)

# ---------------------------------------------------------
# TAB 1: DOCUMENT SCANNER
# ---------------------------------------------------------
with tab1:
    st.header("📷 Document Scanner & Cropper")
    st.caption("Capture a document photo, adjust border cropping using the sliders, and process for shadow removal.")
    
    img_file = st.camera_input("Take a photo of your document")
    
    if img_file:
        bytes_data = img_file.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        height, width, _ = cv_img.shape
        
        st.subheader("✂️ Crop Adjustments")
        col_crop1, col_crop2 = st.columns(2)
        
        with col_crop1:
            top_crop = st.slider("Crop Top (%)", 0, 40, 0, key="top")
            bottom_crop = st.slider("Crop Bottom (%)", 0, 40, 0, key="bottom")
            
        with col_crop2:
            left_crop = st.slider("Crop Left (%)", 0, 40, 0, key="left")
            right_crop = st.slider("Crop Right (%)", 0, 40, 0, key="right")
            
        top_px = int(height * (top_crop / 100))
        bottom_px = int(height * (1 - (bottom_crop / 100)))
        left_px = int(width * (left_crop / 100))
        right_px = int(width * (1 - (right_crop / 100)))
        
        if top_px < bottom_px and left_px < right_px:
            cropped_img = cv_img[top_px:bottom_px, left_px:right_px]
        else:
            cropped_img = cv_img
            
        # LAB Shadow Removal
        lab = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        dilated_l = cv2.dilate(l, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_l, 21)
        diff_img = 255 - cv2.absdiff(l, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        updated_lab = cv2.merge((norm_img, a, b))
        scanned_img = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2RGB)
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.image(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB), caption="Cropped Input Frame", use_container_width=True)
        with col_res2:
            st.image(scanned_img, caption="Processed Output", use_container_width=True)
            
        # Image Pre-processing for Enhanced OCR
        gray = cv2.cvtColor(scanned_img, cv2.COLOR_RGB2GRAY)
        threshold_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Automatic OCR Execution with Optimized Configuration
        try:
            custom_config = r'--oem 3 --psm 6'
            raw_text = pytesseract.image_to_string(threshold_img, config=custom_config)
            cleaned_text = clean_ocr_text(raw_text)
            
            st.subheader("📝 Extracted Text:")
            if cleaned_text.strip():
                st.text_area("OCR Result", cleaned_text, height=200)
                st.download_button("Download Text (.txt)", data=cleaned_text, file_name="scanned_text.txt")
            else:
                st.warning("No clear printed text detected. Ensure proper lighting and printed font.")
        except Exception as e:
            st.error(f"OCR Processing Error: {e}")

# ---------------------------------------------------------
# TAB 2: IMAGE UPLOADER
# ---------------------------------------------------------
with tab2:
    st.header("🖼️ Upload Image File")
    uploaded_image = st.file_uploader("Select an image file", type=["png", "jpg", "jpeg"])
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Document Image", width=400)
        
        # Convert PIL Image to OpenCV format for pre-processing
        open_cv_image = np.array(image.convert('RGB')) 
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
        threshold_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        try:
            custom_config = r'--oem 3 --psm 6'
            raw_text = pytesseract.image_to_string(threshold_img, config=custom_config)
            cleaned_text = clean_ocr_text(raw_text)
            
            st.subheader("📝 Extracted Text:")
            if cleaned_text.strip():
                st.text_area("OCR Result", cleaned_text, height=200)
                st.download_button("Download Text (.txt)", data=cleaned_text, file_name="extracted_text.txt")
            else:
                st.warning("No readable text found in the image.")
        except Exception as e:
            st.error(f"OCR Engine Error: {e}")
