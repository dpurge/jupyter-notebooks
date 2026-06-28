import httpx
import cv2
import numpy as np
import pdf2image
import pytesseract

# from pdf2image import convert_from_path, convert_from_bytes
# from pdf2image.exceptions import (
#     PDFInfoNotInstalledError,
#     PDFPageCountError,
#     PDFSyntaxError
# )

def image_from_url(url):
    with httpx.Client() as client:
        resp = client.get(url)
        resp.raise_for_status()
        image = np.asarray(bytearray(resp.content), dtype=np.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return img

def image_from_pdf(filename, page):
    img = None
    images = pdf2image.convert_from_path(filename, first_page=page, last_page=page)
    if images:
        img_ppm = images[0]
        img_np = np.array(img_ppm)
        img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img

def ocr_image(img, lang):
    text = pytesseract.image_to_string(img, lang=lang)
    return text