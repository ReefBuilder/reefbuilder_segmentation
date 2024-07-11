import fiftyone as fo
import fiftyone.utils.labels as foul
import os

from reefbuilder_segmentation.utils.preprocessor.dataset import (
    preprocess_dataset_with_config,
)


class Preprocessor:
    """
    Preprocessor process the images to create a dataset that is
    ready for the downstream machine learning pipeline
    """

    def __init__(self, image_folder_path, coco_file_paths):
        assert os.path.exists(
            image_folder_path
        ), "Image Folder Path doesnt exist. Please provide new path..."
        self.image_folder_path = image_folder_path

        assert type(coco_file_paths) is list, (
            "COCO files havent been inputted in a list. "
            "Please input a list of COCO files..."
        )
        for file_path in coco_file_paths:
            assert os.path.exists(
                file_path
            ), "At least one COCO file path doesnt exist. Please check..."
        self.coco_file_paths = coco_file_paths
        self.dataset = None
        self.preprocess_config = None
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None

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
