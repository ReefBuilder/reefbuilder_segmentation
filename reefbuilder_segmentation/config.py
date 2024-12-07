import os

# file to store config or global variables

accepted_image_formats = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]
accepted_label_formats = ["json", "JSON"]
base_yolo_model = "yolov8m-seg.pt"
logger_name = "reefbuilder_segmentation"

current_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(current_dir, "logging_config.json")
