import RetinaFace
from retinaface.commons import postprocess

class DetectFace:
    def __init__(self):
        self.model, self.face_detector = self.build_model()

    def build_model(self):
        model = RetinaFace.build_model()
        face_detector = RetinaFace.detect_faces
        return model, face_detector

    def detect_face(self, face_detector, img, align=True):
        resp = []

        obj = RetinaFace.detect_faces(img, model=self.model, threshold=0.9)
        if type(obj) == dict:
            for key in obj:
                identity = obj[key]
                facial_area = identity["facial_area"]

                y = facial_area[1]
                h = facial_area[3] - y
                x = facial_area[0]
                w = facial_area[2] - x
                img_region = [x, y, w, h]

                detected_face = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]

                if align:
                    landmarks = identity["landmarks"]
                    left_eye = landmarks["left_eye"]
                    right_eye = landmarks["right_eye"]
                    nose = landmarks["nose"]

                    detected_face = postprocess.alignment_procedure(detected_face, right_eye, left_eye, nose)

                resp.append((detected_face, img_region))

        return resp
