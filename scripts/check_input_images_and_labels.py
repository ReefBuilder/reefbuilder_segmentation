from reefbuilder_segmentation.checks.images import ImageChecker
from reefbuilder_segmentation.checks.labels import LabelChecker

source_image_folder = "../data/data_sample_correct/images"
source_label_folder = "../data/data_sample_correct/labels"

if __name__ == "__main__":
    ic = ImageChecker(source_image_folder)
    ic.describe()
    ic.check_images()

    lc = LabelChecker(source_label_folder)
    lc.describe()
    lc.check_labels()
