import logging
from ultralytics import YOLO
import os
import shutil

import reefbuilder_segmentation.config as cfg
from reefbuilder_segmentation.utils.modelling.yolo import (
    export_yolo_data,
    test_on_data_with_labels_yolo,
)
from reefbuilder_segmentation.utils.preprocessor.paths import expand_paths


logger = logging.Logger(cfg.logger_name)


class Model:
    """
    Class for creating a model to model the given inputs and train on them.
    """

    def __init__(self, fo_dataset):
        self.dataset = fo_dataset  # fiftyone dataset
        self.model = None  # points to the loaded model itself
        self.model_type = None  # type of model being used: YOLO
        self.model_location = None  # location from where model is loaded
        self.data_folder = None  # folder containing data (when using YOLO)
        self.train_metrics = None
        self.valid_metrics = None  # metrics on validation set
        self.test_metrics = None

        # inference
        self.inference_model_type = None
        self.inference_model_location = None
        self.inference_model = None
        self.inference_image_location = None
        self.inference_output_location = None

    @expand_paths("model_location", "data_location")
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
            logger.info("Directory to create yolo dataset exists. Removing it...")
            shutil.rmtree(data_location)
        logger.info("Creating new yolo dataset directory...")
        os.makedirs(data_location, exist_ok=True)

        # create yolo dataset and files from fiftyone dataset
        self.data_folder = data_location
        splits = ["train", "val", "test"]
        classes = self.dataset.classes["segmentations"]
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
            YOLO(cfg.base_yolo_model)  # this downloads the model

            src = cfg.base_yolo_model
            current_dir = os.path.dirname(os.path.abspath(__file__))
            dst = os.path.join(current_dir, "..", "models", src)
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

    @expand_paths("model_location", "image_location", "output_location")
    def infer_yolo(
        self,
        model_location=None,
        image_location="../data/yolo_files/",
        output_location="../data/output_yolo/",
    ):
        logger.info("Starting inference on images...")
        # todo: clean the below code up into smaller functions
        self.inference_model_type = "YOLO"

        # Ensure the model exists
        if not model_location:
            logger.info("No model location given. Shifting to stored model.")
            if self.model_location:
                model_location = self.model_location
            else:
                logger.error(
                    "No stored model location. Please give valid model location."
                )
                raise FileNotFoundError("Inference model file not found.")

        if not os.path.exists(model_location):
            logger.error("Given inference model location does not exist...")
            raise FileNotFoundError("Inference model file not found.")

        # Load YOLO model
        logger.info("Loading YOLO model for inference...")
        if not model_location:
            self.inference_model_location = model_location
        else:
            self.inference_model_location = self.model_location
        self.inference_model = YOLO(self.inference_model_location)

        # Set up input and output directories
        self.inference_image_location = image_location
        self.inference_output_location = output_location
        os.makedirs(self.inference_output_location, exist_ok=True)
        logger.info(
            f"Inference output folder created at {self.inference_output_location}"
        )

        # todo: run the below through ImageChecker
        # todo: All below like validation things should run through checkers/validators
        image_files = [
            f
            for f in os.listdir(self.inference_image_location)
            if os.path.splitext(f)[1][1:] in cfg.accepted_image_formats
        ]

        if not image_files:
            logger.error("No valid image files found in the input directory.")
            return

        # todo: make this batched inference instead of sequential
        for image_name in image_files:
            image_path = os.path.join(self.inference_image_location, image_name)

            # Perform inference
            results = self.model(image_path)

            # Save inference results
            results_path = os.path.join(
                self.inference_output_location, f"inference_{image_name}"
            )
            results[0].save(filename=results_path)  # This will save annotated images

        logger.info("\n || Inference Complete || \n")
        return None
