# import the necessary packages
import pickle
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from config import config

class Predict:
    """
    A class to predict national card label.

                        Attributes:
                          model:
                            Trained model for classify card images.
                          labels:
                            input card image labels.


                        Methods:
                          load_model(self)
                            A method to load saved model.
                          is_card_valid(self, image: np.ndarray)
                            A method to validate national card.
                          crop_image(self, image: np.ndarray, start_x: int, start_y: int, end_x: int, end_y: int)
                            A method to crop validated image.

                        """

    def __init__(self):
        """
        Initializes class attributes.
                            Args:
                              None

        """
        self.model = None
        self.labels = None

    def load_model(self):
        """This method loads saved model.

                    Args:
                      None
                    Returns:
                      None
                    Raises:
                      IOError: An error occurred loading model.
                    """
        # load our object detector and label binarizer from disk
        print("[INFO] loading object detector...")
        self.model = load_model(config.MODEL_PATH)
        self.labels = pickle.loads(open(config.LB_PATH, "rb").read())

    def is_card_valid(self, image: np.ndarray):
        """This method checks validation of a card.

                    Args:
                      image:
                        input image as a numpy array.


                    Returns:
                      A list containing two objects, one is a flag shows validation of a card and a another a list of
                      bounding box coordinates.

                    Raises:
                      Computational error.
                    """
        (im_height, im_width) = image.shape[:2]
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        (boxPreds, labelPreds) = self.model.predict(image)
        (startX, startY, endX, endY) = boxPreds[0]
        (startX, startY, endX, endY) = (startX * im_width, startY * im_height, endX * im_width, endY * im_height)

        i = np.argmax(labelPreds, axis=1)
        label = self.labels.classes_[i][0]
        if label in ['front_1', 'back_1']:
            return [True, [startX, startY, endX, endY]]
        else:
            return [False, None]

    def crop_image(self, image: np.ndarray, start_x: int, start_y: int, end_x: int, end_y: int):
        """This method checks a directory for duplicate images.

                    Args:
                      image:
                        input image as a numpy array.
                      start_x:
                        start_x coordinate of bounding box.
                      start_y:
                        start_y coordinate of bounding box.
                      end_x:
                        end_x coordinate of bounding box.
                      end_y:
                        end_y coordinate of bounding box.

                    Returns:
                      Cropped Image.

                    Raises:
                      Computational error.
                    """
        start_x = int(start_x)
        start_y = int(start_y)
        end_x = int(end_x)
        end_y = int(end_y)
        return image[start_y:end_y, start_x:end_x]
