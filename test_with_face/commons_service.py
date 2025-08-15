from commons_enums import Models, Metrics, Detector
import warnings
import os
import time
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import commons_distance as dst
import base_NetCust
import commons_functions
from commons_functions import create_resp
from detectors_RWrapper import DetectFace as FaceDetector


class VerificationModel:

    def __init__(self, model_name=Models.FNET512.value, metric=Metrics.COSINE.value, detector=Detector.RFACE.value):
        commons_functions.initialize_folder()
        self.model_name = model_name
        self.model = self.build_model()
        self.metric = metric
        self.detector = detector
        self.face_detector = FaceDetector()

    def build_model(self):
        models = {
            'Facenet512': base_NetCust.loadModel
        }

        model = models.get(self.model_name)
        if model:
            self.model = model()
        else:
            raise ValueError('Invalid model_name passed - {}'.format(self.model_name))

        return self.model

    def verify(self, img1_path, img2_path='', model_name=Models.FNET512.value, distance_metric=Metrics.COSINE.value,
               model=None,
               enforce_detection=False, detector_backend=Detector.RFACE.value, align=True, prog_bar=True,
               normalization='base', is_face=False):
        tic = time.time()

        img_list, bulkProcess = commons_functions.initialize_input(img1_path, img2_path)

        resp_objects = []

        # --------------------------------
        if model_name == 'Ensemble':
            model_names = [Models.FNET512.value]
            metrics = [Metrics.COSINE.value]
        else:
            model_names = [];
            metrics = []
            model_names.append(model_name)
            metrics.append(distance_metric)

        # --------------------------------
        if self.model == None:
            if model_name == 'Ensemble':
                pass
            else:
                self.model = self.build_model()
                models = {}
                models[model_name] = self.model
        else:
            if model_name == 'Ensemble':
                models = self.model.copy()
            else:
                models = {}
                models[model_name] = self.model

        # ------------------------------
        disable_option = (False if len(img_list) > 1 else True) or not prog_bar

        pbar = tqdm(range(0, len(img_list)), desc='Verification', disable=disable_option)

        for index in pbar:

            instance = img_list[index]

            if type(instance) == list and len(instance) >= 2:
                img1_path = instance[0];
                img2_path = instance[1]

                ensemble_features = []

                for i in model_names:
                    custom_model = models[i]
                    img1_representation = self.represent(img_path=img1_path
                                                         , model_name=model_name, model=custom_model
                                                         , enforce_detection=enforce_detection
                                                         ,detector_backend=detector_backend
                                                         ,face_detector=self.face_detector
                                                         , align=align
                                                         , normalization=normalization
                                                         , is_face=False
                                                         )

                    img2_representation = self.represent(img_path=img2_path
                                                         , model_name=model_name, model=custom_model
                                                         , enforce_detection=enforce_detection
                                                         ,detector_backend='skip'
                                                         ,face_detector=self.face_detector
                                                         , align=align
                                                         , normalization=normalization
                                                         , is_face=False
                                                         )

                    for j in metrics:

                        if j == 'cosine':
                            distance = dst.findCosineDistance(img1_representation, img2_representation)
                        elif j == 'euclidean':
                            distance = dst.findEuclideanDistance(img1_representation, img2_representation)
                        elif j == 'euclidean_l2':
                            distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation),
                                                                 dst.l2_normalize(img2_representation))
                        else:
                            raise ValueError("Invalid distance_metric passed - ", distance_metric)

                        distance = np.float64(
                            distance)
                        # ----------------------
                        # decision

                        if model_name != 'Ensemble':

                            threshold = dst.findThreshold(i, j)
                            if distance <= threshold:
                                identified = True
                            else:
                                identified = False

                            resp_obj = {
                                "verified": identified
                                , "distance": distance
                                , "max_threshold_to_verify": threshold
                                , "model": model_name
                                , "similarity_metric": distance_metric

                            }

                            if bulkProcess == True:
                                resp_objects.append(resp_obj)
                            else:
                                return resp_obj

                        else:
                            ensemble_features.append(distance)


            else:
                raise ValueError("Invalid arguments passed to verify function: ", instance)

        # -------------------------
        toc = time.time()

        if bulkProcess == True:

            resp_obj = {}

            for i in range(0, len(resp_objects)):
                resp_item = resp_objects[i]
                resp_obj["pair_%d" % (i + 1)] = resp_item

            return resp_obj

    def represent(self, img_path, model_name=Models.FNET512.value, detector_backend=Detector.RFACE.value, face_detector=None, model=None,
                  enforce_detection=False, align=True, normalization='base', is_face=False):

        if self.model is None:
            self.model = self.build_model()


        input_shape_x, input_shape_y = commons_functions.find_input_shape(self.model)


        img = commons_functions.preprocess_face(img=img_path
                                        , target_size=(input_shape_y, input_shape_x)
                                        , enforce_detection=enforce_detection
                                        , detector_backend=detector_backend
                                        ,face_detector = face_detector
                                        , align=align, is_face=False)


        img = commons_functions.normalize_input(img=img, normalization=normalization)


        embedding = self.model.predict(img, verbose=0)[0].tolist()

        return embedding

    def detectFace(self,img_path, target_size=(224, 224), detector_backend=Detector.RFACE.value, enforce_detection=True,
                   align=True, is_face=False):

        img = commons_functions.preprocess_face(img=img_path, target_size=target_size, detector_backend=detector_backend
                                        , enforce_detection=enforce_detection, align=align, is_face=False)[0]
        return img[:, :, ::-1]

    def verify_face_frames(self, image_source, frames, is_face=False):
        res = {
            'verified': [],
            'distance': []
        }

        for idx, frame in enumerate(frames):
            # print(frame)
            try:

                temp = (self.verify(img1_path=image_source, img2_path=frame,
                                    detector_backend=self.detector, is_face=is_face,
                                    model_name=self.model_name, align=True))

                res['verified'].append(temp['verified'])
                res['distance'].append(temp['distance'])

            except Exception as e:
                res['verified'].append(-1)
                res['distance'].append(-1)

                print(e)

        return res
