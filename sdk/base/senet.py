import tensorflow as tf
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
import yaml
from base.algorithm import Singleton


class Senet(Singleton):

    def get_config(self, yaml_path):
        stream = open(yaml_path, 'r')
        config = yaml.load(stream)
        return config

    def __init__(self, config_path, model_path):
        """
        loads model and image dimension
        """
        self.config = self.get_config(config_path)
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.image_dimension_3D = tuple(self.config['image_dimension_3D'])
        self.detector = MTCNN()

    def reshape_and_normalize_image(self, image):
        """
        turn an image into a proper shape to pass to model
        """
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, tuple(self.config['image_dimension_2D']))
        img = img.reshape(tuple(self.config['image_dimension_4D']))
        img_data = img / 255.

        return img_data

    def predict(self, image1, image2, is_image):
        """
        predicts the similarity of two images
        """
        if not is_image:
            image1 = cv2.imread(str(image1))
            image2 = cv2.imread(str(image2))

        image1 = self.reshape_and_normalize_image(image1)
        image2 = self.reshape_and_normalize_image(image2)

        pairImages = []
        pairImages.append([image1.reshape(self.image_dimension_3D), image2.reshape(self.image_dimension_3D)])
        pairTrain = np.array(pairImages)
        test = [pairTrain[:, 0], pairTrain[:, 1]]
        pred = self.model.predict(test, verbose=0)
        if pred < float(self.config['prediction_threshold']):
            pred_binary = 0
        else:
            pred_binary = 1

        return pred, pred_binary

    def detect_face(self, image, is_image):
        """
        detect faces in an image and choose the face with maximum area
        delete other abnormal images that can not be detected
        """

        face_image = None
        if not is_image:
            image = cv2.imread(image)

        if len(image.shape) == 3:
            a, b, c = image.shape
            # if image has three channels
            if c == 3:
                faces = self.detector.detect_faces(image)
                list_faces = []
                for face in faces:
                    x1, y1, width, height = face['box']
                    x2, y2 = x1 + width, y1 + height
                    face_image = image[y1:y2, x1:x2]
                    (a, b, c) = face_image.shape
                    list_faces.append(a + b)

                    if len(list_faces) > 0:
                        index = list_faces.index(max(list_faces))
                        face = faces[index]
                        x1, y1, width, height = face['box']
                        x2, y2 = x1 + width, y1 + height
                        face_image = image[y1:y2, x1:x2]
                    else:
                        raise Exception('Not found face')

        return face_image

    def predict_with_detect_face(self, image1, image2, is_image):
        """
        first detect faces of images
        then predict their similarity score
        """

        image1 = self.detect_face(image1, is_image)
        image2 = self.detect_face(image2, is_image)
        pred, pred_binary = self.predict(image1, image2, is_image=True)

        return pred, pred_binary

    def predict_with_detect_face_frames(self, frames, image, is_image=True):
        """
        first detect faces of images
        then predict their similarity score
        """
        res = {
            'verified': [],
            'distance': []
        }
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.detect_face(image, is_image)
        for idx, frame in enumerate(frames):
            try:
                # frame = self.detect_face(frame, is_image)
                pred, pred_binary = self.predict(image, frame, is_image=True)

                res['verified'].append(bool(pred_binary))
                res['distance'].append(np.float(pred[0][0]))

            except Exception as e:
                print(e)
                res['verified'].append(-1)

        return res

    def show_image(self, winname, img):
        """
        Shows an image with the window name
        """
        cv2.namedWindow(winname)
        cv2.moveWindow(winname, 40, 30)
        cv2.imshow(winname, img)
        cv2.waitKey()
        cv2.destroyAllWindows()
