import fiftyone.utils.random as four


def preprocess_dataset_with_config(fo_dataset, config):
    # mapping labels
    label_mapping = config['label_mapping']
    new_fo_dataset = fo_dataset.map_labels("segmentations", label_mapping)
    new_fo_dataset = new_fo_dataset.map_labels("detections", label_mapping)
    new_fo_dataset = new_fo_dataset.clone()

    # train, valid, test
    four.random_split(new_fo_dataset, {"train": config["train_percentage"],
                                       "val": config["val_percentage"],
                                       "test": config["test_percentage"]},
                      seed=config['split_seed'])

    train_view, val_view, test_view = (new_fo_dataset.match_tags("train"),
                                       new_fo_dataset.match_tags("val"),
                                       new_fo_dataset.match_tags("test"))
    return train_view.clone(), val_view.clone(), test_view.clone()
