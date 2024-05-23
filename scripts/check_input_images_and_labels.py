from coral_detection.checks.images import ImageChecker
from coral_detection.checks.labels import LabelChecker

source_image_folder = '/Users/ishannangia/github_repos/coral-detection/data/test_data/single_test_images'
source_label_folder = '/Users/ishannangia/github_repos/coral-detection/rough/sample_coco_labels'

if __name__ == "__main__":
    ic = ImageChecker(source_image_folder)
    ic.describe()
    ic.check_images()

    lc = LabelChecker(source_label_folder)
    lc.describe()
    lc.check_labels()
