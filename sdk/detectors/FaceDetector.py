from PIL import Image
import math
import numpy as np
from commons import distance
from base.algorithm import Singleton

class Detector(Singleton):
    def __init__(self):
        pass
    
    def detect_face(self, face_detector, detector_backend, img, align=True):
        obj = self.detect_faces(face_detector, detector_backend, img, align)

        if len(obj) > 0:
            face, region = obj[0]
        else:
            face = None
            region = [0, 0, img.shape[0], img.shape[1]]
        return face, region

    def detect_faces(self, face_detector, detector_backend, img, align=True):

        if face_detector:
            obj = face_detector.detect_face(face_detector, img, align)
            return obj
        else:
            raise ValueError("invalid detector_backend passed - " + detector_backend)

    def alignment_procedure(self, img, left_eye, right_eye):

        left_eye_x, left_eye_y = left_eye
        right_eye_x, right_eye_y = right_eye

        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1


        a = distance.findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
        b = distance.findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
        c = distance.findEuclideanDistance(np.array(right_eye), np.array(left_eye))

        if b != 0 and c != 0:

            cos_a = (b * b + c * c - a * a) / (2 * b * c)
            angle = np.arccos(cos_a)
            angle = (angle * 180) / math.pi

            if direction == -1:
                angle = 90 - angle

            img = Image.fromarray(img)
            img = np.array(img.rotate(direction * angle))

        return img
