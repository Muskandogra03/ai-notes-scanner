import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import io
import img2pdf

st.set_page_config(
    page_title="AI Document Scanner & OCR",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Document Scanner & Multi-Page PDF Exporter")
st.write("Scan document pages, remove shadows, extract OCR text, and generate downloadable PDFs!")

# Initialize Session State for Multi-Page PDF
if 'scanned_pages' not in st.session_state:
    st.session_state.scanned_pages = []

def apply_camscanner_magic_color(img_np):
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    dilated_img = cv2.dilate(l, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(l, bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    enhanced_lab = cv2.merge([norm_img, a, b])
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(enhanced_rgb, -1, kernel)

def apply_clean_bw(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 15)

# Sidebar Setup
st.sidebar.header("⚙️ Scanner Controls")
source = st.sidebar.radio("Select Input Source:", ("Take Photo (Camera)", "Upload Image"))
filter_mode = st.sidebar.selectbox("Filter Effect:", ["Magic Color (CamScanner)", "Clean B&W", "Original Contrast", "Grayscale"])

st.sidebar.subheader("✂️ Crop Adjuster (%)")
top_crop = st.sidebar.slider("Top Crop", 0, 40, 0)
bottom_crop = st.sidebar.slider("Bottom Crop", 0, 40, 0)
left_crop = st.sidebar.slider("Left Crop", 0, 40, 0)
right_crop = st.sidebar.slider("Right Crop", 0, 40, 0)

image_file = st.camera_input("Capture Document Page") if source == "Take Photo (Camera)" else st.file_uploader("Upload Document Photo", type=["jpg", "png", "jpeg"])

if image_file is not None:
    pil_img = Image.open(image_file)
    img_np = np.array(pil_img.convert('RGB'))
    
    h, w, _ = img_np.shape
    t, b = int(h * (top_crop / 100)), int(h * (1 - bottom_crop / 100))
    l, r = int(w * (left_crop / 100)), int(w * (1 - right_crop / 100))
    cropped_np = img_np[t:b, l:r] if (b > t and r > l) else img_np

    if filter_mode == "Magic Color (CamScanner)":
        final_processed = apply_camscanner_magic_color(cropped_np)
    elif filter_mode == "Clean B&W":
        final_processed = apply_clean_bw(cropped_np)
    elif filter_mode == "Original Contrast":
        final_processed = cv2.convertScaleAbs(cropped_np, alpha=1.3, beta=10)
    else:
        final_processed = cv2.cvtColor(cropped_np, cv2.COLOR_RGB2GRAY)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Cropped Original")
        st.image(cropped_np, use_container_width=True)
    with col2:
        st.subheader("2. Scanned Result")
        st.image(final_processed, use_container_width=True)

        # Image Download Button
        res_pil = Image.fromarray(final_processed)
        buf = io.BytesIO()
        res_pil.save(buf, format="PNG")
        st.download_button(
            label="🖼️ Download Scanned Image",
            data=buf.getvalue(),
            file_name="scanned_document.png",
            mime="image/png"
        )

        # Add to PDF Queue Button
        if st.button("➕ Add This Page to PDF"):
            st.session_state.scanned_pages.append(buf.getvalue())
            st.success(f"Page added! Total pages in PDF: {len(st.session_state.scanned_pages)}")

    st.subheader("3. Extracted Text (OCR)")
    try:
        text = pytesseract.image_to_string(final_processed)
        if text.strip():
            st.text_area("OCR Result:", text, height=180)
            st.download_button("📥 Download OCR Text", data=text, file_name="notes_ocr.txt", mime="text/plain")
    except Exception:
        st.error("Tesseract Engine Error during text extraction.")

# PDF Generation Section
if st.session_state.scanned_pages:
    st.markdown("---")
    st.subheader(f"📚 Multi-Page PDF Document ({len(st.session_state.scanned_pages)} Pages)")
    
    col_pdf1, col_pdf2 = st.columns(2)
    with col_pdf1:
        pdf_bytes = img2pdf.convert(st.session_state.scanned_pages)
        st.download_button(
            label="📄 Download Complete PDF",
            data=pdf_bytes,
            file_name="scanned_notes_collection.pdf",
            mime="application/pdf"
        )
    with col_pdf2:
        if st.button("🗑️ Clear PDF Pages"):
            st.session_state.scanned_pages = []
            st.rerun()
