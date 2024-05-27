from coral_detection.preprocess.dataset import Preprocessor

image_folder_path = '/Users/ishannangia/Desktop/del_this_images'
coco_files = ['/Users/ishannangia/Desktop/coco_7.12.22_table_4.json',
              '/Users/ishannangia/Desktop/coco_table5_07_12_22.json']
preprocess_config = {
    "label_mapping": {
        'ref': 'REF',
        'coral': 'CORAL'
    }
}

prep = Preprocessor(image_folder_path, coco_files)
ds = prep.create_dataset()
ds = prep.preprocess_dataset(preprocess_config)
