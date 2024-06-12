from ultralytics import YOLO

import coral_detection.config as cfg
from coral_detection.utils.modelling.yolo import export_yolo_data


class Model:
    def __init__(self, config, train, valid, test):
        self.config = config
        self.train = train
        self.valid = valid
        self.test = test
        self.model = None
        self.model_type = None
        self.model_location = None

    def train_yolo(self, model_location = None, epochs=100, patience=0, **kwargs):
        # create yolo dataset and files from fiftyone dataset
        export_yolo_data(self.train, "../data/train")
        export_yolo_data(self.valid, "../data/valid")
        export_yolo_data(self.test, "../data/test")
        # initialise and read model (pretrained or base)
        if model_location:
            self.model_location = model_location
            try:
                model = YOLO(model_location)
            except:
                print('The given model location doesnt exist...')
                return
        else:
            model = YOLO(cfg.base_yolo_model)
        self.model = model
        # train model
        _ = self.model.train(data="??",
                             epochs=epochs,
                             plots=True,
                             patience=patience,
                             **kwargs)
        # save results on train, validation

    def test_yolo(self, **kwargs):
        # load saved model
        # test on saved test dataset
        # save and return test metrics

    def save_model(self, save_path='./saved_model'):
        # get most recent trained model
        # save in given location
