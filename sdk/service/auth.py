import gc
import os
import cv2
import copy
import json
import base64
import tensorflow as tf
from base.senet import Senet
from commons.errors import Errors
from base.algorithm import Singleton
from backoffice.models import Backoffice
from utils.liveness import LivenessDetection
from utils.verification import FaceVerification, create_resp
from commons.util import make_directory, save_file


class ImageAuth(Singleton):
    def __init__(self):
        self.verify = FaceVerification()
        self.verify_v2 = ''
        if os.getenv("IS_ON") == '1':
            self.verify_v2 = Senet('config/config.yaml', '/opt/1.h5')

        self.liveness = LivenessDetection()

    def get_video_frames_v1(self, video_path, video_len=6):

        capture = cv2.VideoCapture(video_path)
        counter = 0
        frames = []
        boxes = []
        faces = []
        count = 0
        can_rad = False
        while capture.isOpened():
            can_rad = True
            ret, frame = capture.read()
            if ret:
                if counter > video_len:
                    count += 1

                    img_det = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    boxes_ = self.liveness.detector.detect_face(self.liveness.detector.face_detector, img_det,
                                                                align=False)

                    if len(boxes_):
                        for face, box in boxes_:
                            box = [
                                box[0],
                                box[1],
                                box[0] + box[2],
                                box[1] + box[3]
                            ]

                            box[0] = max(box[0], 0)
                            box[1] = max(box[1], 0)


                            if face.shape[0] * face.shape[1] < (100 * 200):
                                continue

                            boxes.append(copy.copy(box))
                            frames.append(img_det)
                            faces.append(copy.copy(face))
                            counter += 1

                    if counter > int(os.getenv("FUM")):
                        break
                    if count > 35:
                        break
                if counter <= video_len:
                    counter += 1
            else:
                break

        del capture, can_rad, count, counter
        return frames, faces, boxes

    def set_detail(self, process, backoffice_res):
        backoffice = Backoffice.objects.get(process=process)
        backoffice.process_res = json.dumps(backoffice_res)
        backoffice.save()

    def get_detail(self, process, typ=Backoffice.Type.AUTH):
        backoffice = Backoffice.objects.get(process=process, process_type=typ)
        return {} if backoffice.process_res == None else json.loads(backoffice.process_res)

    def get_res(self, process, typ=Backoffice.Type.AUTH):
        backoffice = Backoffice.objects.get(process=process, process_type=typ)
        res = json.loads(backoffice.process_res)
        if 'verify_voting_2' in res.keys():
            return res['verify_voting'], res['verify_voting_2'], res['liveness_res']
        return res['verify_voting'], None, res['liveness_res']

    def get_backoffice(self, process):
        return Backoffice.objects.get(process=process)

    def check_verify(self, image, frames, process):
        try:

            backoffice_res = self.get_detail(process)

            res_verification = self.verify.pridect_with_frames(image, frames)

            backoffice_res['verify_res'] = json.dumps(res_verification)

            if len(res_verification['verified']) > 0:
                backoffice_res['verify_voting'] = max(res_verification['verified'],
                                                      key=res_verification['verified'].count)
            else:
                backoffice_res['verify_voting'] = -1

            backoffice_res['verify_status'] = True

            self.set_detail(process, backoffice_res)

            return res_verification
        except Exception as e:
            raise e

    def check_verify_v2(self, image, frames, process):
        try:

            backoffice_res = self.get_detail(process)
            res_verification = self.verify_v2.predict_with_detect_face_frames(frames, image)
            backoffice_res['verify_res_v2'] = json.dumps(res_verification)
            if len(res_verification['verified']) > 0:
                backoffice_res['verify_voting_v2'] = max(res_verification['verified'],
                                                         key=res_verification['verified'].count)

            else:
                backoffice_res['verify_voting_v2'] = -1

            backoffice_res['verify_status_v2'] = True

            self.set_detail(process, backoffice_res)

            return res_verification
        except Exception as e:
            raise e

    def check_liveness(self, process, frames, boxes=None):
        try:
            backoffice_res = self.get_detail(process)

            res_liveness, base64_frame = self.liveness.predict(frames, boxes) \
                if os.getenv("VERSION") == '1' else self.liveness.predict_v2(frames, boxes)

            if len(res_liveness) > 0:
                backoffice_res['liveness_voting'] = max(res_liveness, key=res_liveness.count)

            else:
                backoffice_res['liveness_voting'] = -1

            backoffice_res['liveness_res'] = json.dumps({'ans': res_liveness})
            backoffice_res['liveness_status'] = True

            self.set_detail(process, backoffice_res)
            return res_liveness, base64_frame

        except Exception as e:
            raise e

    def save_file(self, file_path, file_id, is_image=False):
        try:
            data, status = self.cdn.download_file_from_cdn(file_id)
            if status != 200:
                raise ValueError(f'{status}')
            if is_image:
                data = base64.b64decode(data)
            save_file(file_path, data)
        except Exception as e:
            raise e

    def check_file(self, path, data):
        if data != None:
            try:
                self.save_file(path, data)
            except Exception as e:
                raise Exception(e)
        else:
            raise Exception('urn is Null!!!')

    def is_pass_liveness(self, process):
        backoffice = self.get_detail(process)
        votting = backoffice['liveness_voting']
        desc = ''
        name = ''

        if votting == -1:
            desc = Errors.FACE_NOT_FOUND.value
            name = Errors.FACE_NOT_FOUND.name

        elif votting == 0:
            desc = Errors.NOT_LIVE.value
            name = Errors.NOT_LIVE.name

        elif votting == -2:
            desc = Errors.TWO_FACE_FOUND.value
            name = Errors.TWO_FACE_FOUND.name

        elif votting == -3:
            desc = Errors.NOT_STANDARD.value
            name = Errors.NOT_STANDARD.name

        votting = votting if votting else 0
        return desc, votting

    def is_pass_verify(self, process, key):

        backoffice = self.get_detail(process)
        votting = backoffice[key]

        desc = ''
        name = ''

        if votting == False:
            desc = Errors.NOT_MATCH.value
            name = Errors.NOT_MATCH.name
        return desc, votting
