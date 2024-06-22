from reefbuilder_segmentation.preprocess.dataset import Preprocessor
from reefbuilder_segmentation.modelling.base import Model

# preprocessing
image_folder_path = '/Users/ishannangia/Desktop/del_this_images'
coco_files = ['/Users/ishannangia/Desktop/coco_7.12.22_table_4.json']
preprocess_config = {
    "label_mapping": {
        'ref': 'REF',
        'coral': 'CORAL'
    },
    'train_percentage': 0.8,
    'val_percentage': 0.1,
    'test_percentage': 0.1,
    'split_seed': 0
}

prep = Preprocessor(image_folder_path, coco_files)
ds = prep.create_dataset()
prep.preprocess_dataset(preprocess_config)

# modelling
model = Model(prep.dataset)
model.train_yolo(epochs=10,
                 imgsz=640,
                 batch=4)
