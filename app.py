import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ocr_scanner import process_document

st.set_page_config(
    page_title="AI Notes Scanner & OCR",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Notes Document Scanner")
st.write("Scan handwritten/printed notes, auto-crop edges, and extract text instantly!")

st.sidebar.header("Scanner Settings")
source = st.sidebar.radio("Select Input Source:", ("Take Photo (Camera)", "Upload Image"))

image_file = None

if source == "Take Photo (Camera)":
    image_file = st.camera_input("Capture Document Page")
else:
    image_file = st.file_uploader("Upload Document Photo", type=["jpg", "png", "jpeg"])

if image_file is not None:
    st.info("⚡ Processing document using OpenCV Filters & Tesseract OCR...")
    
    outlined, scanned, text = process_document(image_file)

    if outlined is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("1. Edge Detection & Contour")
            st.image(outlined, channels="BGR", use_container_width=True, caption="Detected Green Paper Boundary")

        with col2:
            st.subheader("2. CamScanner Magic Filter")
            st.image(scanned, use_container_width=True, caption="Cropped, Straightened & Clean B&W Document")

        st.subheader("3. Extracted Text Result (OCR)")
        if text.strip():
            st.text_area("Extracted Notes Text:", text, height=250)
            st.download_button(
                label="📥 Download Extracted Text",
                data=text,
                file_name="scanned_notes.txt",
                mime="text/plain"
            )
        else:
            st.warning("No readable text found in the image. Please try capturing with better lighting.")
