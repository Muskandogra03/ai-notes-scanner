import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract

st.set_page_config(
    page_title="AI Document Scanner & OCR",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Professional AI Document Scanner")
st.write("Scan handwritten/printed notes, remove shadows, auto-crop, and apply CamScanner Magic Filters!")

def apply_camscanner_magic_color(img_np):
    # Convert to LAB color space to equalize brightness/remove shadows
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Illumination normalization / shadow removal
    dilated_img = cv2.dilate(l, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(l, bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    # Merge channels back and boost sharp text contrast
    enhanced_lab = cv2.merge([norm_img, a, b])
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    # Subtle sharpening filter for crisp text
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced_rgb, -1, kernel)
    return sharpened

def apply_clean_bw(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    # Smooth background
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Clean adaptive thresholding for clear text without black patches
    bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 15)
    return bw

# Sidebar Controls
st.sidebar.header("⚙️ Scanner Settings")
source = st.sidebar.radio("Select Input Source:", ("Take Photo (Camera)", "Upload Image"))
filter_mode = st.sidebar.selectbox("Select Filter Effect:", ["Magic Color (CamScanner Look)", "Clean B&W (Photocopy Look)", "Original High-Contrast", "Grayscale"])

# Crop Sliders in Sidebar
st.sidebar.subheader("✂️ Manual Crop Adjuster (%)")
top_crop = st.sidebar.slider("Top Edge Crop", 0, 40, 0)
bottom_crop = st.sidebar.slider("Bottom Edge Crop", 0, 40, 0)
left_crop = st.sidebar.slider("Left Edge Crop", 0, 40, 0)
right_crop = st.sidebar.slider("Right Edge Crop", 0, 40, 0)

image_file = None

if source == "Take Photo (Camera)":
    image_file = st.camera_input("Capture Document Page")
else:
    image_file = st.file_uploader("Upload Document Photo", type=["jpg", "png", "jpeg"])

if image_file is not None:
    # Read Image
    pil_img = Image.open(image_file)
    img_np = np.array(pil_img.convert('RGB'))
    
    # Perform Cropping based on Sliders
    h, w, _ = img_np.shape
    t = int(h * (top_crop / 100))
    b = int(h * (1 - bottom_crop / 100))
    l = int(w * (left_crop / 100))
    r = int(w * (1 - right_crop / 100))
    
    # Boundary Safety Check
    if b > t and r > l:
        cropped_np = img_np[t:b, l:r]
    else:
        cropped_np = img_np

    # Apply Selected Filter
    if filter_mode == "Magic Color (CamScanner Look)":
        final_processed = apply_camscanner_magic_color(cropped_np)
    elif filter_mode == "Clean B&W (Photocopy Look)":
        final_processed = apply_clean_bw(cropped_np)
    elif filter_mode == "Original High-Contrast":
        # Boost contrast on original
        alpha = 1.3 # Contrast
        beta = 10   # Brightness
        final_processed = cv2.convertScaleAbs(cropped_np, alpha=alpha, beta=beta)
    else:
        final_processed = cv2.cvtColor(cropped_np, cv2.COLOR_RGB2GRAY)

    # Layout Display
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Cropped Original")
        st.image(cropped_np, use_container_width=True, caption="Adjust edges using sidebar sliders")
        
    with col2:
        st.subheader("2. Scanned Result")
        st.image(final_processed, use_container_width=True, caption=f"Applied Filter: {filter_mode}")
        
    st.subheader("3. Extracted Text (OCR)")
    try:
        text = pytesseract.image_to_string(final_processed)
        if text.strip():
            st.text_area("OCR Result:", text, height=220)
            st.download_button(
                label="📥 Download Extracted Text",
                data=text,
                file_name="scanned_notes.txt",
                mime="text/plain"
            )
        else:
            st.warning("No readable text found. Try adjusting crop sliders or switching filter mode.")
    except Exception as e:
        st.error("Tesseract Engine Error during text extraction.")
