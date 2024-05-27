from coral_detection.preprocess.dataset import Preprocessor

image_folder_path = '/Users/ishannangia/Desktop/del_this_images'
coco_files = ['/Users/ishannangia/Desktop/coco_7.12.22_table_4.json',
              '/Users/ishannangia/Desktop/coco_table5_07_12_22.json']
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
train_view, val_view, test_view = prep.preprocess_dataset(preprocess_config)
print(train_view)
print("\n--\n")
print(val_view)
print("\n--\n")
print(test_view)