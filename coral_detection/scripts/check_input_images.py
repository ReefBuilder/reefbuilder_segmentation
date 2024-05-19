from coral_detection.checks.images import ImageChecker

source_image_folder = '/Users/ishannangia/github_repos/coral-detection/data/test_data/single_test_images'

if __name__ == "__main__":
    ic = ImageChecker(source_image_folder)
    ic.describe()
    ic.check_images()

