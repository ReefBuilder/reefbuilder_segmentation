from fiftyone.utils.image import transform_images


def preprocess_dataset_with_config(fo_dataset, config):
    label_mapping = config['label_mapping']
    new_fo_dataset = fo_dataset.map_labels("segmentations", label_mapping)
    new_fo_dataset = new_fo_dataset.map_labels("detections", label_mapping)
    return new_fo_dataset.clone()

