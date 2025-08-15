from base.net import Network
from base.algorithm import Singleton
from torchvision import transforms
import cv2
import os
import torch
import base64
from PIL import Image
from io import BytesIO
from base.algorithm import BASE_DIR
from base.algorithm import load_dotenv
from detectors.RWrapper import DetectFace

load_dotenv(BASE_DIR / "server/.env")


class LivenessDetection(Singleton):

    def __init__(self):
        self.device = ''

        if os.getenv("ENABLED_T_C") == '1':
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.45, 0)
                self.device = (
                    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                )
            else:
                self.device = (torch.device("cpu"))
        else:
            self.device = (torch.device("cpu"))

        self.checkpoint = torch.load('/opt/net.pth'
                                     , map_location=self.device)

        self.model = Network(False)
        self.model.load_state_dict(self.checkpoint['state_dict'])
        self.model.eval()
        self.model.to(self.device)
        self.detector = DetectFace()

    def frame_to_base46(self, image):
        image = Image.fromarray(image)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        img_str = str(img_str)[2:]
        img_str = img_str[:len(img_str) - 1]
        return img_str

    def check(self):
        used, total = torch.cuda.mem_get_info()
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if used / total < 0.5:
            self.device = (torch.device("cpu"))

    def predict(self, frames, boxes=None, detect=False):
        ans = []
        ans_2 = []
        has_face = False
        face_base64 = ''
        if os.getenv("IS_LIVE") == '0':
            if len(frames):
                RGB = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
                face_base64 = self.frame_to_base46(RGB)
                return [1], face_base64
            else:
                return [], face_base64

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        for idx, frame in enumerate(frames):
            if idx >= int(os.getenv("LIMITED")):
                break

            if detect:
                img_det = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes = self.detector.face_detector(img_det)

                if len(boxes) > 1:
                    ans.append(-2)

                elif len(boxes):
                    for box in boxes:
                        box = boxes['face_1']['facial_area']
                        box[0] = max(box[0], 0)
                        box[1] = max(box[1], 0)

                        anti_img = img_det[abs(box[1] - 30):box[3], abs(box[0] - 10):(box[2] + 20)]

                        if anti_img.shape[0] * anti_img.shape[1] < (100 * 200):
                            ans.append(-3)
                            break

                        if not has_face:
                            RGB = cv2.cvtColor(anti_img, cv2.COLOR_BGR2RGB)
                            face_base64 = self.frame_to_base46(RGB)
                            has_face = True

                        anti_img = transform(anti_img)

                        anti_img = anti_img.unsqueeze(0).to(self.device)

                        dec, binary = self.model.forward(anti_img)
                        res = torch.mean(dec).item()

                        ans_2.append(res)
                        if res < 0:
                            ans.append(0)
                        else:
                            ans.append(1)

                        del anti_img, res
                else:
                    ans.append(-1)
            else:
                box = boxes[idx]
                img_det = frame
                img_det = img_det[abs(box[1] - 30):box[3], abs(box[0] - 10):(box[2] + 20)]
                img_det = cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB)
                if not has_face:
                    RGB = cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB)
                    face_base64 = self.frame_to_base46(RGB)
                    has_face = True

                anti_img = transform(img_det)

                anti_img = anti_img.unsqueeze(0).to(self.device)

                dec, binary = self.model.forward(anti_img)
                res = torch.mean(dec).item()

                ans_2.append(res)
                if res < 0:
                    ans.append(0)
                else:
                    ans.append(1)

        return ans, face_base64

    def predict_v2(self, frames, boxes=None, detect=False):
        ans = []
        ans_2 = []
        has_face = False
        face_base64 = ''
        if os.getenv("IS_LIVE") == '0':
            if len(frames):
                RGB = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
                face_base64 = self.frame_to_base46(RGB)
                return [1], face_base64
            else:
                return [], face_base64

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        for idx, frame in enumerate(frames):
            if idx >= int(os.getenv("LIMITED")):
                break

            if detect:
                img_det = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes = self.detector.face_detector(img_det)

                if len(boxes) > 1:
                    ans.append(-2)

                elif len(boxes):
                    for box in boxes:
                        box = boxes['face_1']['facial_area']
                        box[0] = max(box[0], 0)
                        box[1] = max(box[1], 0)

                        anti_img = img_det[abs(box[1] - 30):box[3], abs(box[0] - 10):(box[2] + 20)]

                        if anti_img.shape[0] * anti_img.shape[1] < (100 * 200):
                            ans.append(-3)
                            break

                        if not has_face:
                            RGB = cv2.cvtColor(anti_img, cv2.COLOR_BGR2RGB)
                            face_base64 = self.frame_to_base46(RGB)
                            has_face = True

                        anti_img = transform(anti_img)

                        anti_img = anti_img.unsqueeze(0).to(self.device)

                        dec, binary = self.model.forward(anti_img)
                        res = torch.mean(dec).item()

                        ans_2.append(res)
                        if res < 0:
                            ans.append(0)
                        else:
                            ans.append(1)

                        del anti_img, res
                else:
                    ans.append(-1)
            else:
                box = boxes[idx]
                img_det = frame

                img_det = img_det[abs(box[1] - 30):(box[3] + 20), abs(box[0] - 10):(box[2] + 20)]

                img_det = cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB)

                if not has_face:
                    RGB = cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB)
                    face_base64 = self.frame_to_base46(RGB)
                    has_face = True

                anti_img = transform(img_det)

                anti_img = anti_img.unsqueeze(0).to(self.device)

                dec, binary = self.model.forward(anti_img)
                res = torch.mean(dec).item()

                ans_2.append(res)
                if res < 0:
                    ans.append(0)
                else:
                    ans.append(1)
        return ans, face_base64
