from ultralytics import YOLO

from coral_detection.utils.test_utils import generate_yolo_prediction_images

model_checkpoint = '../models/megalodon_Coral-Detection-CI-22-23-Season-1_best_180.pt'
test_data_folders = ["../data/test_data/22-23 Coral Table Test Images",
                     "../data/test_data/23-24 AR Test Images",
                     "../data/test_data/23-24 Coral Table Test Images"]
results_save_parent_folder_path = "../../results/"

predict_args = {
    'conf': 0.1,
}

model = YOLO(model_checkpoint)
for test_data_folder in test_data_folders:
    generate_yolo_prediction_images(model, predict_args, test_data_folder, results_save_parent_folder_path)
