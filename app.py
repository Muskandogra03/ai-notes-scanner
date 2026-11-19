import streamlit as st
import cv2
import numpy as np
from PIL import Image
from google import genai
import fitz  # PyMuPDF

# Page Configuration
st.set_page_config(page_title="AI Document Scanner & Study Assistant", layout="wide")

st.title("📚 AI Document Scanner & Study Assistant")
st.markdown("Scan handwritten notes, syllabus screenshots, or multi-page PDFs for human-level text extraction, structured study summaries, and interactive Q&A.")

# Initialize Gemini AI Client
@st.cache_resource
def get_gemini_client():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        return genai.Client(api_key=api_key)
    except Exception:
        return None

client = get_gemini_client()

if client is None:
    st.error("⚠️ GEMINI_API_KEY missing! Please add your free Google AI Studio key in Streamlit Cloud Secrets.")
    st.stop()

# Sidebar Controls
st.sidebar.header("⚙️ Scanner Settings")

input_source = st.sidebar.radio(
    "Select Input Source:",
    ("Take Photo (Camera)", "Upload File (Image / PDF)")
)

filter_effect = st.sidebar.selectbox(
    "Select Filter Effect (Image Only):",
    ["Original Color", "Magic Color (CamScanner Look)", "Original Gray", "B&W High Contrast"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("✂️ Manual Crop Adjuster (%)")

top_crop = st.sidebar.slider("Top Edge Crop", 0, 40, 0)
bottom_crop = st.sidebar.slider("Bottom Edge Crop", 0, 40, 0)
left_crop = st.sidebar.slider("Left Edge Crop", 0, 40, 0)
right_crop = st.sidebar.slider("Right Edge Crop", 0, 40, 0)

# Session state to store extracted content for Q&A
if "study_context" not in st.session_state:
    st.session_state.study_context = ""

# ---------------------------------------------------------
# INPUT HANDLING
# ---------------------------------------------------------
cv_img = None
pdf_bytes = None
pdf_name = None

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
            pdf_bytes = uploaded_file.getvalue()
            pdf_name = uploaded_file.name
            st.success(f"📄 PDF Uploaded: {pdf_name}")
        else:
            bytes_data = uploaded_file.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

# ---------------------------------------------------------
# IMAGE PROCESSING & REAL AI EXTRACTION
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

    if filter_effect == "Magic Color (CamScanner Look)":
        lab = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        dilated_l = cv2.dilate(l, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_l, 21)
        diff_img = 255 - cv2.absdiff(l, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        processed_img = cv2.cvtColor(cv2.merge((norm_img, a, b)), cv2.COLOR_LAB2RGB)
    elif filter_effect == "Original Gray":
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    elif filter_effect == "B&W High Contrast":
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        processed_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    else:
        processed_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB), caption="Original / Cropped Frame", use_container_width=True)
    with col2:
        st.image(processed_img, caption=f"Processed ({filter_effect})", use_container_width=True)

    st.markdown("---")
    if st.button("✨ Extract Text & Generate AI Study Summary"):
        with st.spinner("AI Vision Model reading and organizing your notes..."):
            try:
                pil_image = Image.fromarray(processed_img)
                prompt = (
                    "You are an intelligent academic study assistant. Analyze this image carefully "
                    "(whether it is handwritten notes, printed text, or a syllabus table) and provide:\n\n"
                    "### 📝 Extracted Text\n"
                    "(Transcribe all text accurately and cleanly)\n\n"
                    "### 📌 Structured Study Summary\n"
                    "(Organize the key topics, units, or concepts into clear headings and crisp, easy-to-revise bullet points — never cut sentences mid-way.)"
                )
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[pil_image, prompt]
                )
                result_text = response.text
                st.session_state.study_context = result_text
                st.markdown(result_text)
                st.download_button("⬇️ Download AI Notes (.txt)", data=result_text, file_name="ai_study_notes.txt")
            except Exception as e:
                st.error(f"AI Processing Error: {e}")

# ---------------------------------------------------------
# PDF PROCESSING & REAL AI EXTRACTION (Supports Scanned + Digital PDFs)
# ---------------------------------------------------------
elif pdf_bytes is not None:
    st.markdown("---")
    if st.button("✨ Analyze PDF & Generate AI Study Summary"):
        with st.spinner("Reading PDF pages and generating structured summary..."):
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                contents_to_send = []
                extracted_digital_text = []

                for page_num, page in enumerate(doc):
                    text = page.get_text().strip()
                    if text:
                        extracted_digital_text.append(f"--- Page {page_num + 1} ---\n{text}")
                    else:
                        # Scanned/handwritten PDF page -> convert to image for Gemini Vision
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                        img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        contents_to_send.append(img_pil)

                prompt = (
                    "You are an expert study assistant. Analyze the provided PDF pages/text and create:\n"
                    "1. ### 📌 Clean Unit-wise / Topic-wise Summary (well-structured bullet points, complete sentences)\n"
                    "2. ### 💡 Key Exam Takeaways (most important concepts to remember)"
                )

                if extracted_digital_text:
                    full_text = "\n\n".join(extracted_digital_text)
                    contents_to_send.insert(0, full_text)

                contents_to_send.append(prompt)

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents_to_send
                )
                result_text = response.text
                st.session_state.study_context = result_text
                st.markdown(result_text)
                st.download_button("⬇️ Download PDF Summary (.txt)", data=result_text, file_name="ai_pdf_summary.txt")
            except Exception as e:
                st.error(f"PDF AI Error: {e}")

# ---------------------------------------------------------
# BONUS: INTERACTIVE AI STUDY DOUBT SOLVER
# ---------------------------------------------------------
if st.session_state.study_context:
    st.markdown("---")
    st.subheader("💬 Ask AI About These Notes")
    user_q = st.text_input("Ask a question (e.g., 'Explain Unit 2 simply' or 'Create 5 viva questions from this'):")
    if st.button("Ask AI") and user_q.strip():
        with st.spinner("Thinking..."):
            try:
                qa_prompt = f"Based on these study notes:\n\n{st.session_state.study_context}\n\nAnswer this student's question clearly:\n{user_q}"
                qa_response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=qa_prompt
                )
                st.markdown(qa_response.text)
            except Exception as e:
                st.error(f"Error: {e}")
