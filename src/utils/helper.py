import cv2
import numpy as np
import supervision as sv
from glob import glob
import os

# TODO: read global variables from config file
POSSIBLE_IMAGE_EXTENSIONS = ["jpg", "JPG", "png", "PNG"]


# TODO: add data types to function parameters and function return types
def load_image(img_name, img_address, resized_image_size):
    # load single image
    image_bgr = cv2.imread(img_address)
    original_image_size = image_bgr.shape[1], image_bgr.shape[0]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image_rgb, resized_image_size)  # for inference in YOLO

    return image_bgr, image_rgb, resized_image, original_image_size


def draw_bboxes_xyxyn(bboxes, img):
    colors = [(150, 150, 150)]
    drawn_img = img.copy()
    for i, box in enumerate(bboxes):
        x, y, x1, y1 = box
        x, x1 = x * img.shape[1], x1 * img.shape[1]
        y, y1 = y * img.shape[0], y1 * img.shape[0]
        cv2.rectangle(drawn_img, (int(x), int(y)), (int(x1), int(y1)), colors[0], 10)
    return drawn_img


def get_sam_masks(yolo_result, sam, resized_image):
    # multiple bounding boxes as input for a single image
    input_boxes = yolo_result.boxes.xyxy
    class_ids = yolo_result.boxes.cls.cpu().numpy()

    mask_predictor = SamPredictor(sam)
    transformed_boxes = mask_predictor.transform.apply_boxes_torch(input_boxes, resized_image.shape[:2])
    mask_predictor.set_image(resized_image)
    masks, iou_predictions, low_res_masks = mask_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False
    )
    return masks, class_ids


def create_detections(masks, class_ids):
    # creating Detections object for all the masks
    xyxys = np.array([sv.mask_to_xyxy(masks=i.cpu()) for i in masks])
    xyxys = xyxys.squeeze(1)
    numpy_masks = masks.cpu().numpy().squeeze(1)
    detections = sv.Detections(
        class_id=class_ids,
        xyxy=xyxys,
        mask=numpy_masks
    )
    return detections


def draw_masks_image(image_bgr, detections):
    # bounding boxes and segmented areas
    box_annotator = sv.BoxAnnotator(color=sv.Color.red(), thickness=10)
    mask_annotator = sv.MaskAnnotator(color=sv.Color.red())
    source_image = image_bgr.copy()
    segmented_image = image_bgr.copy()

    source_image = box_annotator.annotate(scene=source_image,
                                          detections=detections,
                                          skip_label=False)
    segmented_image = mask_annotator.annotate(scene=segmented_image,
                                              detections=detections)

    # plot_grid = sv.plot_images_grid(
    #       images=[source_image, segmented_image],
    #       grid_size=(1, 2),
    #       titles=['image with SAM BB', 'segmented image'],
    #       size=(20, 20)
    #   )
    return segmented_image


def get_images_from_folder(folder):
    all_images = []
    for extension in POSSIBLE_IMAGE_EXTENSIONS:
        current_extension_images = glob(os.path.join(folder, f"*.{extension}"))
        all_images.extend(current_extension_images)
    return all_images


def load_image_with_resizing(image_address, resize_image_size):
    # load single image
    image_bgr = cv2.imread(image_address)
    original_image_size = image_bgr.shape[1], image_bgr.shape[0]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image_rgb, resize_image_size) # for inference in YOLO

    return image_bgr, image_rgb, resized_image, original_image_size


# TODO: check cv2 or pillow and then save accordingly
def save_image(image, image_save_path):
    cv2.imwrite(image_save_path, image)
    return None
