import fiftyone as fo
import yaml
from ultralytics import YOLO
import os


# TODO: Extend function to add fo functionalities for customisation in the future
def export_yolo_data(
    samples,
    export_dir=".",
    classes=None,
    label_field=None,
    split=None
):
    if type(split) is list:
        splits = split
        for split in splits:
            export_yolo_data(
                samples,
                export_dir,
                classes,
                label_field,
                split
            )
    else:
        if split is None:
            split_view = samples
            split = "val"
        else:
            split_view = samples.match_tags(split)

        split_view.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="polylines",
            classes=classes,
            split=split,
        )


def generate_test_yaml(yaml_location, ds_split, save_location):
    with open(yaml_location, 'r') as stream:
        try:
            yaml_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        yaml_file["val"] = yaml_file[ds_split]
    with open(save_location, 'w') as stream:
        yaml.dump(yaml_file, stream)
    return


# TODO: modify function to allow testing on any arbitrary dataset with labels
def test_on_data_with_labels_yolo(source_yaml_location,
                                  ds_split=None,
                                  model=None,
                                  model_location=None,
                                  **kwargs):
    # load saved model
    if model is None:
        assert model_location
        model = YOLO(model_location)

    # test on saved test dataset
    test_metrics = model.val(data=source_yaml_location,
                             split=ds_split,
                             **kwargs)

    # save metrics
    return test_metrics
