import cv2
import os


def all_images_same_dim(all_images, dim, assertion_size):
    messages = []
    for image_path in all_images:
        image = cv2.imread(image_path)
        dim_to_assert = image.shape[dim]
        if dim_to_assert != assertion_size:
            messages.append(
                f"Dimension {dim} of {os.path.basename(image_path)} isnt {assertion_size}"
            )
    return messages


def basic_image_check(image_path):
    all_messages = []
    message, image = open_image(image_path)
    all_messages.append(message)
    if not message:
        all_messages.append(n_channels(image, 3, image_path))
        all_messages.append(height_width(image, image_path))
    all_messages = [i for i in all_messages if i is not None]
    return all_messages


def open_image(path):
    message = None
    image = None
    try:
        image = cv2.imread(path)
        assert image is not None
    except:
        message = f"{os.path.basename(path)} is not a valid image file"
    return message, image


def n_channels(image, n, path):
    message = None
    channels = image.shape[-1]
    if channels != n:
        message = f"{os.path.basename(path)} doesnt have {n} channels"
    return message


def height_width(image, path):
    message = None
    height, width, _ = image.shape
    if (height <= 0) or (width <= 0):
        message = f"{os.path.basename(path)} has either h/w == 0"
    return message
