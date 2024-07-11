from reefbuilder_segmentation.preprocess.dataset import Preprocessor
from reefbuilder_segmentation.modelling.base import Model

# preprocessing
image_folder_path = "../data/data_sample_correct/images"
coco_files = ["../data/data_sample_correct/labels/labels-coco.json"]

preprocess_config = {
    "label_mapping": {"eyes": "EYE", "nose": "NOSE"},
    "train_percentage": 0.8,
    "val_percentage": 0.1,
    "test_percentage": 0.1,
    "split_seed": 0,
}

prep = Preprocessor(image_folder_path, coco_files)
ds = prep.create_dataset()
prep.preprocess_dataset(preprocess_config)

# modelling
model = Model(prep.dataset)
model.train_yolo(epochs=10, imgsz=640, batch=4)
