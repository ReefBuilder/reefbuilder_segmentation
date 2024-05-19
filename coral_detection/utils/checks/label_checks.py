import os
import json


def duplicate_image_names(coco_file_paths):
    message = None
    all_image_names = []
    for coco_file in coco_file_paths:
        with open(coco_file, 'r') as f:
            json_file = json.load(f)
        for image_dict in json_file['images']:
            image_name = image_dict['file_name']
            all_image_names.append(image_name)
    n_duplicate_names = len(all_image_names) - len(set(all_image_names))
    if n_duplicate_names:
        duplicate_list = [all_image_names.count(i) for i in all_image_names]
        duplicate_dict = {i: j for i, j in zip(all_image_names, duplicate_list) if j > 1}
        duplicate_names = list(duplicate_dict.keys())
        # TODO: Print out the coco files to check too if duplicates are found
        message = (f'{n_duplicate_names} duplicate image names detected in COCO files.\n'
                   f'Duplicates found for files: {duplicate_names}.')
    return message


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