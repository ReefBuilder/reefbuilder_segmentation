import os
from glob import glob

from coral_detection.utils.general import get_images_from_folder, load_image_with_resizing, save_image
from coral_detection.utils.visualisation_utils import draw_bboxes_xyxyn

# TODO: read global variables from central cfg file
RESIZE_IMAGE_SIZE = (640, 490)


# TODO: make function more modular + allow single test file input
def generate_yolo_prediction_images(model, predict_args, test_folder_path, save_parent_path):
    # creating save location
    test_folder_name = test_folder_path.strip("/").split('/')[-1]
    full_save_path = os.path.join(save_parent_path, test_folder_name)
    if not os.path.exists(full_save_path):
        os.mkdir(full_save_path)

    # Reading in all images
    image_paths, image_bgrs, image_rgbs, resized_images = read_images_from_folder(test_folder_path,
                                                                                  RESIZE_IMAGE_SIZE)
    image_names = [i.split("/")[-1] for i in image_paths]

    # Getting YOLO predictions for multiple images
    results = model.predict(resized_images, **predict_args)
    images_with_bboxes = [draw_bboxes_xyxyn(result.boxes.xyxyn, image_rgb)
                          for result, image_rgb in zip(results, image_rgbs)]
    for image_index, image in enumerate(images_with_bboxes):
        image_save_path = os.path.join(full_save_path, image_names[image_index])
        save_image(image, image_save_path)
    return None


# TODO: make resize_size an optional argument
# TODO: Fix resizing to maintain aspect ratio
def read_images_from_folder(folder, resize_size):
    image_paths = get_images_from_folder(folder)
    image_bgrs = []
    image_rgbs = []
    resized_images = []
    for i_image_index, i_img_address in enumerate(image_paths):
        image_bgr, image_rgb, resized_image, _ = load_image_with_resizing(i_img_address,
                                                                          resize_size)
        image_bgrs.append(image_bgr)
        image_rgbs.append(image_rgb)
        resized_images.append(resized_image)
    return image_paths, image_bgrs, image_rgbs, resized_images
