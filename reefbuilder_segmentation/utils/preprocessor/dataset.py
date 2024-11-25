import fiftyone.utils.random as four
import operator
import logging
import reefbuilder_segmentation.config as cfg
import cv2
import glob
import os
import fiftyone as fo

# Mapping strings to operator functions
string_to_operator = {
    "lt": operator.lt,  # Less than
    "lte": operator.le,  # Less than or equal to
    "gt": operator.gt,  # Greater than
    "gte": operator.ge,  # Greater than or equal to
    "eq": operator.eq,  # Equal
    "neq": operator.ne,  # Not equal
}

logger = logging.Logger(cfg.logger_name)


def preprocess_dataset_with_config(fo_dataset, config):
    # todo: make label name changing easier and more flexible as a function
    # mapping labels and updating classes
    label_mapping = config["label_mapping"]
    new_fo_dataset = fo_dataset.map_labels("segmentations", label_mapping)
    new_fo_dataset = new_fo_dataset.map_labels("detections", label_mapping)
    new_classes = list(set([i for i in label_mapping.values()]))
    new_fo_dataset.classes["segmentations"] = new_classes
    new_fo_dataset.classes["detections"] = new_classes
    new_fo_dataset.save()
    # new_fo_dataset = new_fo_dataset.clone()

    # train, valid, test
    # TODO: Change these in the future to make the model more flexible
    assert_fraction(config["train_fraction"], "gt", "lt")
    assert_fraction(config["val_fraction"], "gt", "lt")
    assert_fraction(config["test_fraction"], "gt", "lt")

    four.random_split(
        new_fo_dataset,
        {
            "train": config["train_fraction"],
            "val": config["val_fraction"],
            "test": config["test_fraction"],
        },
        seed=config["split_seed"],
    )

    train_view, val_view, test_view = (
        new_fo_dataset.match_tags("train"),
        new_fo_dataset.match_tags("val"),
        new_fo_dataset.match_tags("test"),
    )
    return (
        new_fo_dataset,
        train_view.clone(),
        val_view.clone(),
        test_view.clone(),
    )


def assert_fraction(frac, lower_boundary_condition, upper_boundary_condition):
    try:
        lower_operator = string_to_operator[lower_boundary_condition]
        upper_operator = string_to_operator[upper_boundary_condition]
        assert lower_operator(frac, 0) and upper_operator(
            frac, 1
        ), "Please input a proper fraction greater than 0 and less than 1"
    except AssertionError as e:
        # Log the assertion error
        logger.error(f"Assertion failed: {e}")
        # Optionally, re-raise the exception if you want the program to stop
        raise
    return None


def read_write_images_cv2(list_of_image_file_paths):
    # this helps prevent corrupted images warning
    for image_path in list_of_image_file_paths:
        # Load the image
        image = cv2.imread(image_path)
        # Save the image
        cv2.imwrite(image_path, image)
    return


def create_inference_fo_dataset(image_folder_path):
    # Create samples for your data
    samples = []
    for filepath in glob.glob(os.path.join(image_folder_path, "*")):
        sample = fo.Sample(filepath=filepath)
        samples.append(sample)

    # Create dataset
    dataset = fo.Dataset("inference-dataset")
    dataset.add_samples(samples)
    return dataset


def create_train_fo_dataset(image_folder_path, coco_file_paths):
    coco_dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=image_folder_path,
        labels_path=coco_file_paths[0],
    )
    for coco_path in coco_file_paths[1:]:
        coco_dataset.merge_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=image_folder_path,
            labels_path=coco_path,
        )
    return coco_dataset
