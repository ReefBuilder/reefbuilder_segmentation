import fiftyone as fo
import fiftyone.utils.labels as foul
import os
import glob
import logging
import reefbuilder_segmentation.config as cfg

from reefbuilder_segmentation.utils.preprocessor.dataset import (
    preprocess_dataset_with_config,
    read_write_images_cv2,
)
from reefbuilder_segmentation.utils.preprocessor.paths import expand_paths

logger = logging.Logger(cfg.logger_name)


class Preprocessor:
    """
    Preprocessor process the images to create a dataset that is
    ready for the downstream machine learning pipeline
    """

    @expand_paths("image_folder_path", "label_folder_path")
    def __init__(self, image_folder_path, label_folder_path=None):
        assert os.path.exists(
            image_folder_path
        ), "Image Folder Path doesnt exist. Please provide new path..."

        if label_folder_path is not None:
            assert os.path.exists(
                label_folder_path
            ), "Label Folder Path doesnt exist. Please provide new path..."

        self.image_folder_path = image_folder_path
        self.label_folder_path = label_folder_path
        self.coco_file_paths = glob.glob(os.path.join(label_folder_path, "*"))
        self.dataset = None
        self.preprocess_config = None
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.mode = None

        # todo: might want to shift this to imagechecker
        # to avoid corrupt jpeg images error from ultralytics
        read_write_images_cv2(glob.glob(os.path.join(image_folder_path, "*")))

        # set mode basis input arguments
        if image_folder_path and label_folder_path:
            self.mode = "train"
        elif image_folder_path and (not label_folder_path):
            self.mode = "inference"
        logger.info(f"Preprocessor built for mode: {self.mode}")

    def create_dataset(self):
        coco_dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=self.image_folder_path,
            labels_path=self.coco_file_paths[0],
        )
        for coco_path in self.coco_file_paths[1:]:
            coco_dataset.merge_dir(
                dataset_type=fo.types.COCODetectionDataset,
                data_path=self.image_folder_path,
                labels_path=coco_path,
            )
        self.dataset = coco_dataset
        return self.dataset

    def preprocess_dataset(self, config):
        self.preprocess_config = config
        datasets = preprocess_dataset_with_config(self.dataset, config)
        self.dataset, self.train_dataset, self.val_dataset, self.test_dataset = datasets
        # TODO: check and modify tolerance if needed
        foul.instances_to_polylines(
            self.dataset, "segmentations", "polylines", tolerance=2
        )
        return self.train_dataset, self.val_dataset, self.test_dataset
