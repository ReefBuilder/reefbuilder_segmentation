import fiftyone as fo


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
