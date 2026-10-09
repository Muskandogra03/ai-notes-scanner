import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract

# Page Configuration
st.set_page_config(page_title="AI Document Scanner", layout="wide")

st.title("📚 AI Document Scanner & Image OCR Engine")
st.markdown("Scan printed documents with real-time shadow removal and manual cropping, or extract text from uploaded images.")

# Navigation Tabs
tab1, tab2 = st.tabs([
    "📷 Document Scanner", 
    "🖼️ Image Uploader"
])

def process_image_for_ocr(image_np):
    """Enhances image contrast for better printed text OCR extraction."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Gaussian Blur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu's Binarization
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# ---------------------------------------------------------
# TAB 1: DOCUMENT SCANNER
# ---------------------------------------------------------
with tab1:
    st.header("📷 Document Scanner & Cropper")
    st.caption("Capture a printed document photo, adjust border cropping using the sliders, and process for shadow removal.")
    
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
            
        # LAB Shadow Removal Algorithm
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
            
        # Image Pre-processing for OCR
        thresh_img = process_image_for_ocr(scanned_img)
        
        # OCR Processing
        try:
            custom_config = r'--oem 3 --psm 6'
            raw_text = pytesseract.image_to_string(thresh_img, config=custom_config)
            
            st.subheader("📝 Extracted Text:")
            if raw_text.strip():
                st.text_area("OCR Result", raw_text, height=200)
                st.download_button("Download Text (.txt)", data=raw_text, file_name="scanned_text.txt")
            else:
                st.warning("No readable printed text detected. Please capture a clear printed document.")
        except Exception as e:
            st.error(f"OCR Engine Error: {e}")

# ---------------------------------------------------------
# TAB 2: IMAGE UPLOADER
# ---------------------------------------------------------
with tab2:
    st.header("🖼️ Upload Image File")
    uploaded_image = st.file_uploader("Select an image file", type=["png", "jpg", "jpeg"])
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Document Image", width=400)
        
        open_cv_image = np.array(image.convert('RGB')) 
        thresh_img = process_image_for_ocr(open_cv_image)
        
        try:
            custom_config = r'--oem 3 --psm 6'
            raw_text = pytesseract.image_to_string(thresh_img, config=custom_config)
            
            st.subheader("📝 Extracted Text:")
            if raw_text.strip():
                st.text_area("OCR Result", raw_text, height=200)
                st.download_button("Download Text (.txt)", data=raw_text, file_name="extracted_text.txt")
            else:
                st.warning("No readable printed text found in the image.")
        except Exception as e:
            st.error(f"OCR Engine Error: {e}")
