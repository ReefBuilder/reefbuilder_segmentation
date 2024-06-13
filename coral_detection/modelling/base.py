from ultralytics import YOLO
import os

import coral_detection.config as cfg
from coral_detection.utils.modelling.yolo import export_yolo_data, test_on_data_with_labels_yolo


class Model:
    """
    Class for creating a model to model the given inputs and train on them.
    """
    def __init__(self, fo_dataset):
        self.dataset = fo_dataset  # fiftyone dataset
        self.model = None  # points to the loaded model itself
        self.model_type = None  # type of model currently being used: YOLO, DETECTRON
        self.model_location = None  # location from where model has been loaded
        self.data_folder = None  # folder containing the actual data (when using YOLO)
        self.train_metrics = None
        self.valid_metrics = None  # metrics on validation set
        self.test_metrics = None

    def train_yolo(self, model_location=None, data_location='../data', epochs=100, patience=0, **kwargs):
        self.model_type = 'YOLO'

        # create yolo dataset and files from fiftyone dataset
        self.data_folder = data_location
        export_yolo_data(self.dataset, self.data_folder, split=['train', 'val', 'test'])

        # initialise and read model (pretrained or base)
        if model_location:
            self.model_location = model_location
            try:
                model = YOLO(model_location)
            except:
                print('The given model location doesnt exist...')
                return
        else:
            # TODO: manage downloading of model. Should be placed in a correct folder
            # TODO: update model_location
            print('Starting training from base model...')
            model = YOLO(cfg.base_yolo_model)
        self.model = model

        # TODO: manage run folder creation. Where should it get created?
        # train model
        results = self.model.train(data=os.path.join(self.data_folder, 'dataset.yaml'),
                                   epochs=epochs,
                                   plots=True,
                                   patience=patience,
                                   **kwargs)

        # updating model parameters
        self.model_location = os.path.join(results.save_dir, 'weights', 'best.pt')
        self.model = YOLO(self.model_location)

        # updating results
        self.train_metrics = test_on_data_with_labels_yolo(os.path.join(self.data_folder, 'dataset.yaml'),
                                                           'train',
                                                           model=self.model)
        self.valid_metrics = test_on_data_with_labels_yolo(os.path.join(self.data_folder, 'dataset.yaml'),
                                                           'val',
                                                           model=self.model)
        self.test_metrics = test_on_data_with_labels_yolo(os.path.join(self.data_folder, 'dataset.yaml'),
                                                          'test',
                                                          model=self.model)

        print('\n || Model results saved here:', results.save_dir, "|| \n")
        return None


