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

def save_image(img, filename):
    cv2.imwrite(filename, img)

def invert_image(img):
    return cv2.bitwise_not(img)

def grayscale_image(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def binarize_image(img, thresh=127, maxval=255):
    gray = grayscale_image(img)
    _, black_white = cv2.threshold(gray, thresh, maxval, cv2.THRESH_BINARY)
    return black_white

def thin_font(img):
    img = cv2.bitwise_not(img)
    kernel = np.ones((2,2), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.bitwise_not(img)
    return img

def thick_font(img):
    img = cv2.bitwise_not(img)
    kernel = np.ones((2,2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.bitwise_not(img)
    return img

def despecle_image(img):
    new_img = img.copy()
    dilate_kernel = np.ones((1,1), np.uint8)
    new_img = cv2.dilate(new_img, dilate_kernel, iterations=1)
    erode_kernel = np.ones((1,1), np.uint8)
    new_img = cv2.erode(new_img, erode_kernel, iterations=1)
    new_img = cv2.morphologyEx(new_img, cv2.MORPH_CLOSE, erode_kernel)
    new_img = cv2.medianBlur(new_img, 3)
    return new_img

def get_skew_angle(img) -> float:
    new_img = img.copy()

    gray_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (9,9), 0)
    thresh_img = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(thresh_img, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # for c in contours:
    #     rect = cv2.boundingRect(c)
    #     x, y, w, h = rect
    #     cv2.rectangle(new_img, (x,y), (x+w, y+h), (0, 255, 0), 2)

    largest_contour = contours[0]
    min_area_rect = cv2.minAreaRect(largest_contour)

    angle = min_area_rect[-1]
    if angle < -45:
        angle = angle + 90

    # return -1.0 * angle, new_img
    return float(angle)

def rotate_image(img, angle: float):
    new_img = img.copy()
    (h, w) = new_img.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_img = cv2.warpAffine(new_img, m, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return new_img

def deskew_image(img):
    angle = get_skew_angle(img)
    return rotate_image(img, angle)

def remove_border(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    contour = contours_sorted[-1]
    x, y, w, h = cv2.boundingRect(contour)
    crop = img[y:y+h, x:x+w]
    return crop

def add_border(img, color=[255,255,255], top=100, bottom=100, left=100, right=100):
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_img

def draw_boxes(img, boxes, color=(36, 255, 12), width=2):
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(img, (x,y), (x+w, y+h), color, width)

def get_boxes(img, blurring_box=(7, 7), structuring_box = (3, 13), iterations=1, filter=lambda b: True):
    new_img = img.copy()
    gray_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, blurring_box, 0)
    thresh_img = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, structuring_box)
    dilated = cv2.dilate(thresh_img, kernel, iterations=iterations)

    contours = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    boxes = []
    for c in contours:
        box = cv2.boundingRect(c)
        if filter(box):
            boxes.append(box)

    return boxes

    # contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    # draw_boxes(new_img, boxes=contours)