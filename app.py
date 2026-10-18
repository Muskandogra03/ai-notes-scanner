import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
import fitz  # PyMuPDF
import re
from collections import Counter

# Page Configuration
st.set_page_config(page_title="AI Document Scanner & Study Assistant", layout="wide")

st.title("📚 AI Document Scanner & Study Assistant")
st.markdown("Scan printed documents, handwritten notes, or upload multi-page PDF files for free text extraction and summary generation.")

# Initialize EasyOCR Reader (Cached for performance)
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'], gpu=False)

with st.spinner("Loading Deep Learning OCR Engine..."):
    reader = load_ocr_reader()

# Sidebar Controls
st.sidebar.header("⚙️ Scanner Settings")

# Input Source Selection
input_source = st.sidebar.radio(
    "Select Input Source:",
    ("Take Photo (Camera)", "Upload File (Image / PDF)")
)

# Filter Selection
filter_effect = st.sidebar.selectbox(
    "Select Filter Effect (Image Only):",
    ["Magic Color (CamScanner Look)", "Adaptive Handwriting Enhancement", "Original Gray", "B&W High Contrast"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("✂️ Manual Crop Adjuster (%)")

top_crop = st.sidebar.slider("Top Edge Crop", 0, 40, 0)
bottom_crop = st.sidebar.slider("Bottom Edge Crop", 0, 40, 0)
left_crop = st.sidebar.slider("Left Edge Crop", 0, 40, 0)
right_crop = st.sidebar.slider("Right Edge Crop", 0, 40, 0)

def enhance_handwriting(img_np):
    """Deep image pre-processing pipeline for handwritten text."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    
    # Bilateral filter removes background paper noise while keeping edges sharp
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    return enhanced

def generate_extractive_summary(text, max_sentences=5):
    """Generates a dynamic summary based on sentence relevance."""
    sentences = re.split(r'\.\s+|\n+', text)
    clean_sentences = [s.strip() for s in sentences if len(s.strip()) > 12]
    
    if not clean_sentences:
        return []

    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    stop_words = set(["the", "and", "for", "that", "this", "with", "from", "are", "was", "been", "using", "have", "has"])
    filtered_words = [w for w in words if w not in stop_words]
    word_counts = Counter(filtered_words)

    sentence_scores = {}
    for i, sentence in enumerate(clean_sentences):
        score = 0
        for word in re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower()):
            if word in word_counts:
                score += word_counts[word]
        sentence_scores[i] = score

    top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:max_sentences]
    top_indices.sort()
    
    summary = [clean_sentences[idx] for idx in top_indices]
    return summary

# ---------------------------------------------------------
# INPUT HANDLING
# ---------------------------------------------------------
cv_img = None
pdf_text_content = None

if input_source == "Take Photo (Camera)":
    img_file = st.camera_input("Capture Document or Handwritten Notes")
    if img_file:
        bytes_data = img_file.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

else:
    uploaded_file = st.file_uploader("Upload Image or PDF Document", type=["png", "jpg", "jpeg", "pdf"])
    if uploaded_file:
        file_type = uploaded_file.name.split(".")[-1].lower()
        
        if file_type == "pdf":
            st.success(f"📄 PDF Document Uploaded: {uploaded_file.name}")
            with st.spinner("Extracting text from PDF pages..."):
                try:
                    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                    extracted_pages = []
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        extracted_pages.append(page.get_text())
                    pdf_text_content = "\n".join(extracted_pages)
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
        else:
            bytes_data = uploaded_file.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

# ---------------------------------------------------------
# IMAGE PROCESSING PIPELINE
# ---------------------------------------------------------
if cv_img is not None:
    height, width, _ = cv_img.shape
    
    top_px = int(height * (top_crop / 100))
    bottom_px = int(height * (1 - (bottom_crop / 100)))
    left_px = int(width * (left_crop / 100))
    right_px = int(width * (1 - (right_crop / 100)))
    
    if top_px < bottom_px and left_px < right_px:
        cropped_img = cv_img[top_px:bottom_px, left_px:right_px]
    else:
        cropped_img = cv_img

    # Filter Applications
    if filter_effect == "Magic Color (CamScanner Look)":
        lab = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        dilated_l = cv2.dilate(l, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_l, 21)
        diff_img = 255 - cv2.absdiff(l, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        updated_lab = cv2.merge((norm_img, a, b))
        processed_img = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2RGB)

    elif filter_effect == "Adaptive Handwriting Enhancement":
        enhanced_gray = enhance_handwriting(cropped_img)
        processed_img = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)

    elif filter_effect == "Original Gray":
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    else:
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        processed_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB), caption="Original / Cropped Frame", use_container_width=True)
    with col2:
        st.image(processed_img, caption=f"Processed ({filter_effect})", use_container_width=True)

    # OCR Section
    st.markdown("---")
    st.subheader("📝 Extracted Text & Summary")
    
    if st.button("🔍 Extract Text & Read Notes"):
        with st.spinner("Running Neural OCR Engine on Handwritten Content..."):
            try:
                # Pass contrast enhanced frame to EasyOCR
                enhanced_frame = enhance_handwriting(cropped_img)
                results = reader.readtext(enhanced_frame, detail=0, paragraph=True)
                ocr_text = "\n".join(results)
                
                if ocr_text.strip():
                    st.text_area("Extracted OCR Text", ocr_text, height=220)
                    
                    st.markdown("---")
                    st.markdown("### 📌 Document Summary")
                    summary_points = generate_extractive_summary(ocr_text)
                    
                    if summary_points:
                        for point in summary_points:
                            st.markdown(f"* {point}")
                    else:
                        st.info("Summary generated from full extracted text above.")
                        
                    st.download_button("Download Extracted Text (.txt)", data=ocr_text, file_name="scanned_notes.txt")
                else:
                    st.warning("No readable text detected. Ensure proper lighting and select 'Adaptive Handwriting Enhancement' filter.")
            except Exception as e:
                st.error(f"OCR Extraction Error: {e}")

# ---------------------------------------------------------
# PDF PROCESSING PIPELINE
# ---------------------------------------------------------
elif pdf_text_content is not None:
    st.markdown("---")
    st.subheader("📝 Extracted PDF Text & Summary")
    
    if pdf_text_content.strip():
        st.text_area("PDF Extracted Text", pdf_text_content, height=250)
        
        st.markdown("---")
        st.markdown("### 📌 PDF Summary")
        summary_points = generate_extractive_summary(pdf_text_content)
        
        if summary_points:
            for point in summary_points:
                st.markdown(f"* {point}")
        else:
            st.info("Full extracted text displayed above.")
            
        st.download_button("Download PDF Text (.txt)", data=pdf_text_content, file_name="pdf_extracted_notes.txt")
    else:
        st.warning("No selectable text found in the PDF.")
