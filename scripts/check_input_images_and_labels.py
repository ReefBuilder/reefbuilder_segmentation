from reefbuilder_segmentation.checks.images import ImageChecker
from reefbuilder_segmentation.checks.labels import LabelChecker

source_image_folder = "/Users/ishannangia/github_repos/reefbuilder_segmentation/data/test_data/single_test_images"
source_label_folder = (
    "/Users/ishannangia/github_repos/reefbuilder_segmentation/rough/sample_coco_labels"
)

if __name__ == "__main__":
    ic = ImageChecker(source_image_folder)
    ic.describe()
    ic.check_images()

    lc = LabelChecker(source_label_folder)
    lc.describe()
    lc.check_labels()
