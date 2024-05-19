import cv2
import os


def open_image(path):
    message = None
    image = None
    try:
        image = cv2.imread(path)
        assert image is not None
    except:
        message = f'{os.path.basename(path)} is not a valid image file'
    return message, image
