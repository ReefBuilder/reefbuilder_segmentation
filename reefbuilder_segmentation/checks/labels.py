from glob import glob
import os
import logging
import json
import reefbuilder_segmentation.config as cfg
from reefbuilder_segmentation.utils.checks import label_checks
from reefbuilder_segmentation.utils.preprocessor.paths import validate_and_expand_paths

logger = logging.getLogger(cfg.logger_name)


class LabelChecker:
    """
    Implements basic checks on a folder of files with labels
     to ensure that downstream processing is not a problem
    """

    @validate_and_expand_paths("source_folder_path")
    def __init__(self, source_folder_path):
        """
        Initialise the function with a path to the source
        folder where all label files are located
        """
        # saving and storing images
        self.source_folder_path = os.path.expanduser(source_folder_path)
        self.source_labels = []
        for label_format in cfg.accepted_label_formats:
            this_path = os.path.join(
                self.source_folder_path, f"*.{label_format}"
            )  # noqa
            labels = glob(this_path)
            labels = [os.path.abspath(label) for label in labels]
            self.source_labels.extend(labels)

    def describe(self):
        """
        Provides a basic description of the label files
        contained in the folder
        """
        logger.info("Describing image labels now...")
        logger.info(f"Number of files: {len(self.source_labels)}")

    def check_labels(self):
        logger.info(
            "Validating image labels now. Warning messages will appear if error cases are encountered..."
        )
        for file in self.source_labels:
            with open(file, "r") as f:
                json_file = json.load(f)
            messages = label_checks.coco_validator(json_file, file)
            for message in messages:
                logger.warning(message)
        message = label_checks.duplicate_image_names(self.source_labels)
        if message:
            logger.warning(message)
        return None
