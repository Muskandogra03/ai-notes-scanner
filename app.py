import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import fitz  # PyMuPDF (PDF reading ke liye)
from youtube_transcript_api import YouTubeTranscriptApi

st.set_page_config(page_title="AI Document Scanner & Study Assistant", layout="wide")

st.title("📚 AI Study Assistant & Document Scanner")

# Tab navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📷 Document Scanner", 
    "🖼️ Image Uploader",
    "📄 PDF Uploader", 
    "🎥 YouTube Summarizer", 
    "❓ Ask Questions"
])

# Global memory (kisi bhi tab ka text Q&A mein chalega)
if "extracted_text" not in st.session_state:
    st.session_state["extracted_text"] = ""

# ---------------------------------------------------------
# TAB 1: DOCUMENT SCANNER (Live Camera)
# ---------------------------------------------------------
with tab1:
    st.header("📷 Live Document Scanner (Shadow Removal)")
    img_file = st.camera_input("Photo Click Karein")
    
    if img_file:
        bytes_data = img_file.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # LAB Color Space Shadow Removal
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        dilated_l = cv2.dilate(l, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_l, 21)
        diff_img = 255 - cv2.absdiff(l, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        updated_lab = cv2.merge((norm_img, a, b))
        scanned_img = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2RGB)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv_img, caption="Original Image", use_container_width=True)
        with col2:
            st.image(scanned_img, caption="Clean Scanned Image", use_container_width=True)
            
        text = pytesseract.image_to_string(scanned_img)
        st.session_state["extracted_text"] = text
        st.success("Text extract ho gaya! Ab 'Ask Questions' tab par ja kar sawal pooch sakte hain.")

# ---------------------------------------------------------
# TAB 2: IMAGE UPLOADER (Fixed!)
# ---------------------------------------------------------
with tab2:
    st.header("🖼️ Upload Image (PNG / JPG / JPEG)")
    uploaded_image = st.file_uploader("Gallery se Image select karein", type=["png", "jpg", "jpeg"])
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Document Image", width=400)
        
        if st.button("Extract Text from Image"):
            with st.spinner("OCR Processing chal raha hai..."):
                ocr_text = pytesseract.image_to_string(image)
                st.session_state["extracted_text"] = ocr_text
                st.text_area("Extracted Text from Image:", ocr_text, height=200)
                st.success("Image Text Save Ho Gaya! Ask Questions tab par questions poochna shuru karein.")

# ---------------------------------------------------------
# TAB 3: PDF UPLOADER
# ---------------------------------------------------------
with tab3:
    st.header("📄 Upload PDF Document")
    uploaded_pdf = st.file_uploader("PDF File select karein", type=["pdf"])
    
    if uploaded_pdf:
        doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
        pdf_text = ""
        for page in doc:
            pdf_text += page.get_text()
        st.session_state["extracted_text"] = pdf_text
        st.text_area("Extracted Text from PDF:", pdf_text, height=200)
        st.success("PDF Text Save Ho Gaya!")

# ---------------------------------------------------------
# TAB 4: YOUTUBE LINK SUMMARIZER
# ---------------------------------------------------------
with tab4:
    st.header("🎥 YouTube Video Notes & Summary")
    yt_url = st.text_input("YouTube Video URL Paste Karein:")
    
    if yt_url:
        try:
            if "v=" in yt_url:
                video_id = yt_url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in yt_url:
                video_id = yt_url.split("youtu.be/")[1].split("?")[0]
            else:
                video_id = None
                
            if video_id:
                st.video(yt_url)
                if st.button("Generate Video Summary"):
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                    full_transcript = " ".join([item['text'] for item in transcript])
                    st.session_state["extracted_text"] = full_transcript
                    
                    st.subheader("📝 Video Summary Notes:")
                    st.write(full_transcript[:1000] + "...")
                    st.success("Video Transcript Save Ho Gayi!")
        except Exception:
            st.error("Is video ki transcript/subtitles available nahi hain.")

# ---------------------------------------------------------
# TAB 5: ASK QUESTIONS & AUTO SUMMARY
# ---------------------------------------------------------
with tab5:
    st.header("❓ Ask Questions from Loaded Content")
    context = st.session_state.get("extracted_text", "")
    
    if not context:
        st.warning("Pehle kisi bhi tab (Scanner, Image, PDF, ya YouTube) se content load karein!")
    else:
        if st.button("📌 Generate Bullet Points Summary"):
            sentences = [s.strip() for s in context.split(".") if len(s.strip()) > 15]
            st.subheader("Summary Key Points:")
            for pt in sentences[:5]:
                st.write(f"• {pt}")
                
        st.markdown("---")
        user_q = st.text_input("Apna sawal poochein:")
        if user_q:
            words = user_q.lower().split()
            matched = [s.strip() for s in context.split(".") if any(w in s.lower() for w in words if len(w) > 3)]
            st.subheader("🤖 Answer:")
            if matched:
                for m in matched[:3]:
                    st.write(f"👉 ...{m}...")
            else:
                st.write("Sawal ka jawab text me nahi mila.")
