import io
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
from ocr_scanner import scan_and_extract

st.set_page_config(page_title="Document Scanner Module", page_icon="📑")

st.title("📑 Smart Document Scanner Module")
st.write("MCA Mini Project - Phase 1 Presentation")

st.markdown("---")

# 1. Input Source Selection
input_mode = st.radio(
    "Choose Input Source:",
    ["📷 Live Mobile Camera", "📁 Upload Image or PDF"],
    horizontal=True,
)

image_to_process = None

# 2. Input Handling based on choice
if input_mode == "📷 Live Mobile Camera":
    camera_file = st.camera_input("Take a photo of your document")
    if camera_file is not None:
        image_to_process = camera_file

elif input_mode == "📁 Upload Image or PDF":
    uploaded_file = st.file_uploader(
        "Upload Notes Image (JPG/PNG) or PDF Document",
        type=["jpg", "jpeg", "png", "pdf"],
    )

    if uploaded_file is not None:
        # Check if uploaded file is PDF
        if uploaded_file.name.lower().endswith(".pdf"):
            st.info("Converting PDF Page 1 to Image for scanning...")
            try:
                # Convert first page of PDF to Image
                pdf_images = convert_from_bytes(uploaded_file.read())
                first_page = pdf_images[0]

                # Convert PIL Image back to bytes for OpenCV scanner module
                img_byte_arr = io.BytesIO()
                first_page.save(img_byte_arr, format="JPEG")
                image_to_process = img_byte_arr.getvalue()

            except Exception as e:
                st.error(
                    f"PDF Processing Error: {str(e)}. (Agar Poppler set nahi hai, toh JPG/PNG images use karein)"
                )
        else:
            image_to_process = uploaded_file

# 3. Display Image & Perform OCR Scanning
if image_to_process is not None:
    st.image(image_to_process, caption="Original Input Document", width=400)

    if st.button("🔍 Scan & Extract Text", type="primary"):
        with st.spinner("Processing image through OpenCV & PyTesseract..."):
            extracted_text, processed_img, error = scan_and_extract(
                image_to_process
            )

            if error:
                st.error(f"Error: {error}")
            else:
                st.success("Scanning & OCR Completed Successfully!")

                # Column layout for processed image and output text
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("OpenCV Processed Image")
                    st.image(
                        processed_img,
                        caption="Thresholded Image for OCR",
                        use_container_width=True,
                    )

                with col2:
                    st.subheader("Extracted Text")
                    st.text_area(
                        "Result:",
                        extracted_text,
                        height=300,
                        disabled=False,  # Set to False so users can copy/edit text
                    )