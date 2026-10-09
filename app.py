import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
from collections import Counter

# Page Configuration
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

st.sidebar.markdown("---")
st.sidebar.subheader("✂️ Manual Crop Adjuster (%)")

top_crop = st.sidebar.slider("Top Edge Crop", 0, 40, 0)
bottom_crop = st.sidebar.slider("Bottom Edge Crop", 0, 40, 0)
left_crop = st.sidebar.slider("Left Edge Crop", 0, 40, 0)
right_crop = st.sidebar.slider("Right Edge Crop", 0, 40, 0)

def generate_extractive_summary(text, max_sentences=4):
    """Generates a dynamic summary based on sentence relevance in the extracted OCR text."""
    # Split text into sentences or meaningful blocks
    sentences = re.split(r'\.\s+|\n+', text)
    clean_sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    
    if not clean_sentences:
        return []

    # Calculate word frequency across the document
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    # Exclude common stop words
    stop_words = set(["the", "and", "for", "that", "this", "with", "from", "are", "was", "been", "using", "have", "has"])
    filtered_words = [w for w in words if w not in stop_words]
    word_counts = Counter(filtered_words)

    # Score each sentence based on word frequency
    sentence_scores = {}
    for i, sentence in enumerate(clean_sentences):
        score = 0
        for word in re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower()):
            if word in word_counts:
                score += word_counts[word]
        sentence_scores[i] = score

    # Select top-ranked sentences while preserving original order
    top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:max_sentences]
    top_indices.sort()
    
    summary = [clean_sentences[idx] for idx in top_indices]
    return summary

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

    # Filter Application
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

    else:
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        processed_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    # Display Side-by-Side Images
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB), caption="Original / Cropped Frame", use_container_width=True)
    with col2:
        st.image(processed_img, caption=f"Processed ({filter_effect})", use_container_width=True)

    # Pre-processing for OCR
    gray_ocr = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    ocr_input = cv2.resize(gray_ocr, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    # OCR Section
    st.markdown("---")
    st.subheader("📝 Extracted Text & Summary")
    
    try:
        custom_config = r'--oem 3 --psm 3'
        ocr_text = pytesseract.image_to_string(ocr_input, lang='eng', config=custom_config)
        
        if ocr_text.strip():
            st.text_area("OCR Result Output", ocr_text, height=200)
            
            # Dynamic Summary Generator Button
            if st.button("📌 Generate Summary from Uploaded Document"):
                st.markdown("### 📌 Extracted Key Summary Points")
                summary_points = generate_extractive_summary(ocr_text)
                
                if summary_points:
                    for point in summary_points:
                        st.markdown(f"* {point}")
                else:
                    st.warning("Could not form a meaningful summary. Please ensure the document contains clear readable text.")
                    
            st.download_button("Download Text File (.txt)", data=ocr_text, file_name="scanned_notes.txt")
        else:
            st.warning("No text detected. Try adjusting crop parameters or capturing a clearer printed image.")
            
    except Exception as e:
        st.error(f"OCR Engine Error: {e}")
