from ultralytics import YOLO

from utils.test_utils import generate_yolo_prediction_images

model_checkpoint = '../models/coral-detection-best-174.pt'
test_data_folders = ["../data/test_data/22-23 Coral Table Test Images",
                     "../data/test_data/23-24 AR Test Images",
                     "../data/test_data/23-24 Coral Table Test Images"]
results_save_parent_folder_path = "../results/"

model = YOLO(model_checkpoint)
for test_data_folder in test_data_folders:
    generate_yolo_prediction_images(model, test_data_folder, results_save_parent_folder_path)
