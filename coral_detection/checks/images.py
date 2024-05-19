from glob import glob
import cv2
import numpy as np
import os
import coral_detection.config as cfg
from coral_detection.utils.general import print_unique_count_of_arrays
from coral_detection.utils import checks


class ImageChecker:
    """
    Implements basic checks on a folder of images to ensure that downstream processing is not a problem
    """
    def __init__(self, source_folder_path):
        """
        Initialise the function with a path to the source folder where all images are located
        """
        # saving and storing images
        self.source_folder_path = source_folder_path
        self.source_images = []
        for image_format in cfg.accepted_image_formats:
            images = glob(os.path.join(self.source_folder_path, f"*.{image_format}"))
            images = [os.path.abspath(image) for image in images]
            self.source_images.extend(images)

        # setting up default checks
        self.check_if_all_images_same_height = False
        self.check_if_all_images_same_width = False

    def describe(self):
        """
        Provides a basic description of the images contained in the folder
        """
        print(f'Number of files:', len(self.source_images))
        formats = []
        heights = []
        widths = []
        channels = []
        # TODO: should we make this exif data based to reduce loading?
        # TODO: Abstract this function and make a common describe_images function
        for image_path in self.source_images:
            image_format = image_path.split(".")[-1]
            image = cv2.imread(image_path)
            height, width, n_channels = image.shape

            formats.append(image_format)
            heights.append(height)
            widths.append(width)
            channels.append(n_channels)
        print_unique_count_of_arrays([np.array(formats),
                                      np.array(heights),
                                      np.array(widths),
                                      np.array(channels)],
                                     ['- Extension', '- Height', '- Width', '- Number of Channels'])

    # TODO: add logger support for below function
    def check_images(self):
        for image_path in self.source_images:
            messages = checks.basic_image_check(image_path)
            for message in messages:
                print(message)
        if self.check_if_all_images_same_height:
            messages = checks.all_images_same_dim(self.source_images, 1, self.check_if_all_images_same_height)
            for message in messages:
                print(message)
        if self.check_if_all_images_same_width:
            messages = checks.all_images_same_dim(self.source_images, 2, self.check_if_all_images_same_width)
            for message in messages:
                print(message)
        return None
