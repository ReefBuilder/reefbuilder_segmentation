from glob import glob
import os
import coral_detection.config as cfg
from coral_detection.utils.checks import label_checks


class LabelChecker:
    """
    Implements basic checks on a folder of files with labels to ensure that downstream processing is not a problem
    """
    def __init__(self, source_folder_path):
        """
        Initialise the function with a path to the source folder where all label files are located
        """
        # saving and storing images
        self.source_folder_path = source_folder_path
        self.source_labels = []
        for label_format in cfg.accepted_label_formats:
            labels = glob(os.path.join(self.source_folder_path, f"*.{label_format}"))
            labels = [os.path.abspath(label) for label in labels]
            self.source_labels.extend(labels)

    def describe(self):
        """
        Provides a basic description of the label files contained in the folder
        """
        print(f'Number of files:', len(self.source_labels))

    # TODO: add logger support for below function
    def check_labels(self):
        return None
