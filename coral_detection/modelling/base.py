from ultralytics import YOLO
import os

import coral_detection.config as cfg
from coral_detection.utils.modelling.yolo import export_yolo_data


class Model:
    def __init__(self, fo_dataset):
        self.dataset = fo_dataset
        self.model = None
        self.model_type = None
        self.model_location = None
        self.data_folder = None
        self.valid_metrics = None
        self.model_type = None

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

        # save results on train, validation
        self.valid_metrics = self.model.val()
        self.model = os.path.join(results.save_dir, 'weights', 'best.pt')
        print('Model results saved here:', results.save_dir)
        return None

    def test_yolo(self, **kwargs):
        # load saved model
        # test on saved test dataset
        # save and return test metrics
        return

    def save_model(self, save_path='./saved_model'):
        # get most recent trained model
        # save in given location
        return
