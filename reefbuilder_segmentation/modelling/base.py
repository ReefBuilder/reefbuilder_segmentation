import logging
from ultralytics import YOLO
import os
import shutil

import reefbuilder_segmentation.config as cfg
from reefbuilder_segmentation.utils.modelling.yolo import (
    export_yolo_data,
    test_on_data_with_labels_yolo,
)

logger = logging.Logger(cfg.logger_name)


class Model:
    """
    Class for creating a model to model the given inputs and train on them.
    """

    def __init__(self, fo_dataset):
        self.dataset = fo_dataset  # fiftyone dataset
        self.model = None  # points to the loaded model itself
        self.model_type = None  # type of model being used: YOLO, DETECTRON
        self.model_location = None  # location from where model is loaded
        self.data_folder = None  # folder containing data (when using YOLO)
        self.train_metrics = None
        self.valid_metrics = None  # metrics on validation set
        self.test_metrics = None

    def train_yolo(
        self,
        model_location=None,
        data_location="../data/yolo_files/",
        epochs=100,
        patience=0,
        **kwargs,
    ):
        self.model_type = "YOLO"

        # ensuring data dir is created fresh
        if os.path.exists(data_location):
            shutil.rmtree(data_location)
        os.makedirs(data_location, exist_ok=True)

        # create yolo dataset and files from fiftyone dataset
        self.data_folder = data_location
        splits = ["train", "val", "test"]
        classes = self.dataset.default_classes
        export_yolo_data(self.dataset, self.data_folder, split=splits, classes=classes)

        # initialise and read model (pretrained or base)
        if model_location:
            self.model_location = model_location
            try:
                model = YOLO(model_location)
            except:
                logger.error("The given model location doesnt exist...")
                return
        else:
            # TODO: allow for specifying location for saving downloaded model
            logger.info("Starting training from base model...")
            model = YOLO(cfg.base_yolo_model)

            src = cfg.base_yolo_model
            dst = os.path.join("..", "models", src)
            parent_dst = os.path.split(dst)[0]
            if not os.path.exists(parent_dst):
                os.makedirs(parent_dst)
            shutil.move(src, dst)
            model = YOLO(dst)
            self.model_location = dst

        self.model = model

        # TODO: manage run folder creation. Where should it get created?
        # train model
        results = self.model.train(
            data=os.path.join(self.data_folder, "dataset.yaml"),
            epochs=epochs,
            plots=True,
            patience=patience,
            **kwargs,
        )

        # updating model parameters
        loc = os.path.join(results.save_dir, "weights", "best.pt")
        self.model_location = loc
        self.model = YOLO(self.model_location)

        # updating results
        data_yaml = os.path.join(self.data_folder, "dataset.yaml")
        self.train_metrics = test_on_data_with_labels_yolo(
            data_yaml, "train", model=self.model
        )
        self.valid_metrics = test_on_data_with_labels_yolo(
            data_yaml, "val", model=self.model
        )
        self.test_metrics = test_on_data_with_labels_yolo(
            data_yaml, "test", model=self.model
        )

        logger.info(f"\n || Model results saved here: {results.save_dir} || \n")  # noqa
        print("\n || Model results saved here:", results.save_dir, "|| \n")
        return None
