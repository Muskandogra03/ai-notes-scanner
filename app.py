import streamlit as st
import cv2
import numpy as np
from PIL import Image
from google import genai
from google.genai import types
import fitz  # PyMuPDF
import io
import time

st.set_page_config(page_title="AI Document Scanner & Study Assistant", layout="wide")

st.title("📚 AI Document Scanner & Study Assistant")
st.markdown("Scan handwritten notes, syllabus screenshots, or multi-page PDFs for fast text extraction, structured study summaries, and interactive Q&A.")

@st.cache_resource
def get_gemini_client():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=25000)
        )
    except Exception:
        return None

client = get_gemini_client()

if client is None:
    st.error("⚠️ GEMINI_API_KEY missing! Please add your key in Streamlit Cloud Secrets.")
    st.stop()

FAST_CONFIG = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_budget=0),
    temperature=0.2,
    max_output_tokens=1400
)

@st.cache_resource(show_spinner=False)
def get_working_models(_client):
    preferred_order = [
        "gemini-3.8-flash",
        "gemini-3.8-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.0-flash",
        "gemini-2.5-flash",
    ]
    found = []
    try:
        for m in _client.models.list():
            name = getattr(m, "name", "")
            short_name = name.replace("models/", "")
            if short_name and "embedding" not in short_name and "imagen" not in short_name and "veo" not in short_name:
                found.append(short_name)
    except Exception:
        pass

    ordered = []
    for p in preferred_order:
        if p in found and p not in ordered:
            ordered.append(p)
    for f in found:
        if ("flash" in f) and f not in ordered:
            ordered.append(f)
    for f in found:
        if f not in ordered:
            ordered.append(f)

    return ordered if ordered else ["gemini-3.8-flash"]

def compress_image_to_jpeg_part(pil_img, max_dim=800, quality=72):
    w, h = pil_img.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")

def generate_fast_response(contents):
    models_to_try = get_working_models(client)
    last_err = None

    for model_name in models_to_try[:5]:
        try:
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=FAST_CONFIG
            )
        except Exception as e:
            last_err = e
            err_str = str(e)
            if "404" in err_str or "NOT_FOUND" in err_str:
                continue
            if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                time.sleep(0.4)
                continue
            raise e

    raise last_err

st.sidebar.header("⚙️ Scanner Settings")

with st.sidebar.expander("🔍 Available Models Check"):
    st.write(get_working_models(client))

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

if "study_context" not in st.session_state:
    st.session_state.study_context = ""

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
        start_time = time.time()
        with st.spinner("Extracting and summarizing..."):
            try:
                img_part = compress_image_to_jpeg_part(Image.fromarray(processed_img))
                prompt = (
                    "Analyze this document/note image and provide:\n"
                    "### 📝 Extracted Text\n"
                    "(Clean, accurate transcription)\n\n"
                    "### 📌 Structured Study Summary\n"
                    "(Organize topics/units into clear headings and crisp, complete bullet points)"
                )
                response = generate_fast_response([img_part, prompt])
                elapsed = round(time.time() - start_time, 2)
                result_text = response.text
                st.session_state.study_context = result_text
                st.success(f"⚡ Completed in {elapsed} seconds!")
                st.markdown(result_text)
                st.download_button("⬇️ Download AI Notes (.txt)", data=result_text, file_name="ai_study_notes.txt")
            except Exception as e:
                elapsed = round(time.time() - start_time, 2)
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    st.warning(f"⚠️ Gemini servers are temporarily experiencing high traffic ({elapsed}s). Please click the button again in a few seconds.")
                else:
                    st.error(f"AI Processing Error ({elapsed}s): {e}")

elif pdf_bytes is not None:
    st.markdown("---")
    if st.button("✨ Analyze PDF & Generate AI Study Summary"):
        start_time = time.time()
        with st.spinner("Reading PDF..."):
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                contents_to_send = []
                extracted_digital_text = []

                for page_num, page in enumerate(doc[:5]):
                    text = page.get_text().strip()
                    if text:
                        extracted_digital_text.append(f"--- Page {page_num + 1} ---\n{text}")
                    else:
                        pix = page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False)
                        img_part = compress_image_to_jpeg_part(
                            Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        )
                        contents_to_send.append(img_part)

                prompt = (
                    "Analyze the provided PDF content and create:\n"
                    "1. ### 📌 Clean Unit-wise / Topic-wise Summary (structured bullet points, complete sentences)\n"
                    "2. ### 💡 Key Exam Takeaways (most important points)"
                )

                if extracted_digital_text:
                    contents_to_send.insert(0, "\n\n".join(extracted_digital_text))

                contents_to_send.append(prompt)

                response = generate_fast_response(contents_to_send)
                elapsed = round(time.time() - start_time, 2)
                result_text = response.text
                st.session_state.study_context = result_text
                st.success(f"⚡ Completed in {elapsed} seconds!")
                st.markdown(result_text)
                st.download_button("⬇️ Download PDF Summary (.txt)", data=result_text, file_name="ai_pdf_summary.txt")
            except Exception as e:
                elapsed = round(time.time() - start_time, 2)
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    st.warning(f"⚠️ Gemini servers are temporarily experiencing high traffic ({elapsed}s). Please click the button again in a few seconds.")
                else:
                    st.error(f"PDF AI Error ({elapsed}s): {e}")

if st.session_state.study_context:
    st.markdown("---")
    st.subheader("💬 Ask AI About These Notes")
    user_q = st.text_input("Ask a question (e.g., 'Explain Unit 2 simply' or 'Create 5 viva questions from this'):")
    if st.button("Ask AI") and user_q.strip():
        start_time = time.time()
        with st.spinner("Generating answer..."):
            try:
                qa_prompt = f"Based on these study notes:\n\n{st.session_state.study_context}\n\nAnswer clearly:\n{user_q}"
                qa_response = generate_fast_response(qa_prompt)
                elapsed = round(time.time() - start_time, 2)
                st.caption(f"⚡ Answered in {elapsed}s")
                st.markdown(qa_response.text)
            except Exception as e:
                elapsed = round(time.time() - start_time, 2)
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    st.warning(f"⚠️ Gemini servers are busy ({elapsed}s). Please try asking again in a moment.")
                else:
                    st.error(f"Error ({elapsed}s): {e}")

