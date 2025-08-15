import tensorflow as tf
import numpy as np
import cv2


class Verify:

    def __init__(self, config):
        """
        loads model and image dimensions
        """
        self.config = config
        self.model = tf.keras.models.load_model(self.config['load_model_path'], compile=False)
        self.image_dimension_2D = tuple(self.config['image_dimension_2D'])
        self.image_dimension_3D = tuple(self.config['image_dimension_3D'])
        self.image_dimension_4D = tuple(self.config['image_dimension_4D'])

    def reshape_and_normalize_image(self, image):
        """
        turn an image into a proper shape to pass to model
        :param image: image numpy
        :return: reshaped and resized grayscale image
        """
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_dimension_2D)
        img = img.reshape(self.image_dimension_4D)
        img_data = img / 255.

        return img_data

    def predict(self, image1, image2, is_path):
        """
        predicts the similarity of two images
        :param image1: image 1 path or numpy
        :param image2: image 2 path or numpy
        :param is_path: True or False
        :return: prediction score
        :return: prediction in binary format
        """
        if is_path:
            image1 = cv2.imread(str(image1))
            image2 = cv2.imread(str(image2))
        image1 = self.reshape_and_normalize_image(image1)
        image2 = self.reshape_and_normalize_image(image2)
        pair_images = []
        pair_images.append([image1.reshape(self.image_dimension_3D), image2.reshape(self.image_dimension_3D)])
        pair_train = np.array(pair_images)
        test = [pair_train[:, 0], pair_train[:, 1]]
        pred = self.model.predict(test)
        if pred < float(self.config['prediction_threshold']):
            pred_binary = 0
        else:
            pred_binary = 1
        return pred, pred_binary

    def predict_with_detect_face(self, image1, image2, is_path):
        """
        first detects faces of images
        then predicts their similarity score
        :param image1: image 1 path or numpy
        :param image2: image 2 path or numpy
        :param is_path: True or False
        :return: prediction score
        :return: prediction in binary format
        """
        if is_path:
            image1 = cv2.imread(str(image1))
            image2 = cv2.imread(str(image2))
        image1 = self.detect_face(image1)
        image2 = self.detect_face(image2)
        pred, pred_binary = self.predict(image1, image2, is_path=False)

        return pred, pred_binary






