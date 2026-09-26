import cv2
import numpy as np
import pytesseract
from PIL import Image

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def process_document(image_bytes):
    # Load Image
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return None, None, "Invalid Image File"

    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)

    # Find contours
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    doc_cnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc_cnt = approx
            break

    # Boundary Drawing
    outlined = orig.copy()
    if doc_cnt is not None:
        cv2.drawContours(outlined, [doc_cnt], -1, (0, 255, 0), 5)
        # Perspective Transform (Straighten Paper)
        warped = four_point_transform(orig, doc_cnt.reshape(4, 2))
    else:
        warped = orig.copy()

    # Apply CamScanner-style Magic Color/Threshold Filter
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    scanned_filter = cv2.adaptiveThreshold(
        warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 10
    )

    # OCR Text Extraction
    try:
        extracted_text = pytesseract.image_to_string(scanned_filter)
    except Exception as e:
        extracted_text = "OCR Processing Error: Tesseract Engine Error."

    return outlined, scanned_filter, extracted_text 
    
