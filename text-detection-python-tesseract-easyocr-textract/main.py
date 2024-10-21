import cv2
import numpy as np
import pytesseract
from PIL import Image


def preprocess_image(image):
    # Rasmni kulrang shkala(grayscale)ga o'tkazish
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Shovqinni kamaytirish uchun Gaussian blur qo'llash
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive threshold qo'llash
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh


def extract_text(image):
    # Tesseract OCR konfiguratsiyasi
    config = ('-l eng --oem 1 --psm 3')
    # Rasmdan matnni ajratib olish
    text = pytesseract.image_to_string(image, config=config)
    return text


def main():
    # Pasport rasmini o'qish
    image = cv2.imread('/home/davron/PycharmProjects/text-detection-python-tesseract-easyocr-textract/passport.jpg')


    # Rasmni qayta ishlash
    processed_image = preprocess_image(image)

    # Matnni ajratib olish
    text = extract_text(processed_image)

    # Natijalarni chop etish
    print("Aniqlangan matn:")
    print(text)


if __name__ == "__main__":
    main()