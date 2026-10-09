import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract

# Page Config
st.set_page_config(page_title="AI Document Scanner & OCR", layout="wide")

st.title("📚 AI Document Scanner & OCR Engine")

# Sidebar Controls
st.sidebar.header("⚙️ Scanner Settings")

# Input Source Selection
input_source = st.sidebar.radio(
    "Select Input Source:",
    ("Take Photo (Camera)", "Upload Image")
)

# Filter Selection
filter_effect = st.sidebar.selectbox(
    "Select Filter Effect:",
    ["Magic Color (CamScanner Look)", "Original Gray", "B&W High Contrast"]
)

# OCR Mode Option
ocr_mode = st.sidebar.selectbox(
    "Select OCR Binarization Mode:",
    ["Standard Thresholding (Printed Text)", "Adaptive Thresholding (Handwritten/Low Contrast)"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("✂️ Manual Crop Adjuster (%)")

top_crop = st.sidebar.slider("Top Edge Crop", 0, 40, 5)
bottom_crop = st.sidebar.slider("Bottom Edge Crop", 0, 10, 0)
left_crop = st.sidebar.slider("Left Edge Crop", 0, 40, 0)
right_crop = st.sidebar.slider("Right Edge Crop", 0, 40, 0)

cv_img = None

# Input Processing
if input_source == "Take Photo (Camera)":
    img_file = st.camera_input("Capture Document")
    if img_file:
        bytes_data = img_file.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

else:
    uploaded_file = st.file_uploader("Upload Document Image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        bytes_data = uploaded_file.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

# Main Image Processing Engine
if cv_img is not None:
    height, width, _ = cv_img.shape
    
    # Calculate crop pixel boundaries
    top_px = int(height * (top_crop / 100))
    bottom_px = int(height * (1 - (bottom_crop / 100)))
    left_px = int(width * (left_crop / 100))
    right_px = int(width * (1 - (right_crop / 100)))
    
    if top_px < bottom_px and left_px < right_px:
        cropped_img = cv_img[top_px:bottom_px, left_px:right_px]
    else:
        cropped_img = cv_img

    # Apply Selected Filter for Visual Output
    if filter_effect == "Magic Color (CamScanner Look)":
        lab = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        dilated_l = cv2.dilate(l, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_l, 21)
        diff_img = 255 - cv2.absdiff(l, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        updated_lab = cv2.merge((norm_img, a, b))
        processed_img = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2RGB)

    elif filter_effect == "Original Gray":
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    else:  # B&W High Contrast
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        processed_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    # Display Side-by-Side Images
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB), caption="Original / Cropped Frame", use_container_width=True)
    with col2:
        st.image(processed_img, caption=f"Processed ({filter_effect})", use_container_width=True)

    # Pre-processing Dedicated for OCR Engine
    gray_ocr = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    
    # Upscale Image to improve character recognition resolution
    gray_ocr = cv2.resize(gray_ocr, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    if ocr_mode == "Standard Thresholding (Printed Text)":
        ocr_input = cv2.threshold(gray_ocr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    else:
        ocr_input = cv2.adaptiveThreshold(
            gray_ocr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

    # OCR Section
    st.markdown("---")
    st.subheader("📝 Extracted Text (OCR)")
    
    try:
        # Configuration: OEM 3 (Default LSTM Engine), PSM 6 (Assume uniform text block)
        custom_config = r'--oem 3 --psm 6'
        ocr_text = pytesseract.image_to_string(ocr_input, config=custom_config)
        
        if ocr_text.strip():
            st.text_area("OCR Result Output", ocr_text, height=200)
            st.download_button("Download Text File (.txt)", data=ocr_text, file_name="scanned_notes.txt")
        else:
            st.warning("No clear text detected. Adjust cropping sliders or try switching OCR Binarization Mode.")
    except Exception as e:
        st.error(f"OCR Engine Error: {e}")
