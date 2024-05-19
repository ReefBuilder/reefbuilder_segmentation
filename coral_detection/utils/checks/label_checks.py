import os


def coco_validator(coco, file_path):
    messages = []
    file_base_name = os.path.basename(file_path)
    n_missing_keys = check_coco_keys(coco)
    if n_missing_keys:
        messages.append(f"COCO file {file_base_name} was missing {n_missing_keys} keys in the file")
    try:
        result = check_coco_categories(coco["categories"])
        if result:
            messages.append(f"COCO file {file_base_name} has a problematic categories object")
        result = check_coco_images(coco["images"])
        if result:
            messages.append(f"COCO file {file_base_name} has a problematic images object")
        result = check_coco_annotations(coco["annotations"])
        if result:
            messages.append(f"COCO file {file_base_name} has a problematic annotations object")
    except Exception as e:
        messages.append(f'Error encountered while going through COCO file {file_base_name} with arguments {e.args}')
        pass
    return messages


def check_coco_keys(coco):
    needed_keys = {'info', 'licenses', 'categories', 'images', 'annotations'}
    existing_keys = set(coco.keys())
    missing_keys = len(needed_keys - existing_keys)
    return missing_keys


def check_coco_categories(categories_list):
    needed_keys = {'id', 'name', 'supercategory'}
    ids = []
    for category_dict in categories_list:
        ids.append(int(category_dict['id']))
        existing_keys = set(category_dict.keys())
        missing_keys = len(needed_keys - existing_keys)
        if missing_keys:
            return True
    if (max(ids) + 1) != len(categories_list):
        return True
    return False


def check_coco_images(images_list):
    needed_keys = {'id', 'license', 'file_name', 'height', 'width'}
    ids = []
    for images_dict in images_list:
        ids.append(int(images_dict['id']))
        existing_keys = set(images_dict.keys())
        missing_keys = len(needed_keys - existing_keys)
        if missing_keys:
            return True
    if max(ids) != len(images_list):
        return True
    return False


def check_coco_annotations(annotations_list):
    needed_keys = {'id', 'image_id', 'category_id', 'bbox', 'area', 'segmentation', 'iscrowd'}
    ids = []
    for annotations_dict in annotations_list:
        ids.append(int(annotations_dict['id']))
        existing_keys = set(annotations_dict.keys())
        missing_keys = len(needed_keys - existing_keys)
        if missing_keys:
            return True
    if max(ids) != len(annotations_list):
        return True
    return False