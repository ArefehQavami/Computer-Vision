import copy
import cv2
import os
from utils_liveness import LivenessDetection
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils_verification import FaceVerification
from base_senet import Senet
def get_video_frames_v1(video_path, video_len=6):
    from mtcnn import MTCNN
    detector = MTCNN()
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
                boxes_ = detector.detect_faces(img_det)

                if len(boxes_):
                    for box in boxes_:

                        x1, y1, width, height = box['box']
                        x2, y2 = x1 + width, y1 + height
                        face = img_det[y1:y2, x1:x2]
                        if face.shape[0] * face.shape[1] < (120 * 120):
                            continue
                        boxes.append(copy.copy([x1, y1, x2, y2]))
                        frames.append(img_det)
                        faces.append(copy.copy(face))
                        counter += 1

                if counter > 13:
                    break
                if count > 35:  # 55
                    break
            if counter <= video_len:
                counter += 1
        else:
            break

    return frames, faces, boxes


def main():

    verify = FaceVerification()
    verify_v2 = Senet('E:/Face Verification/senet-face/code/check_liveness/config_config.yaml',
                      'E:/Face Verification/senet-face/code/check_liveness/1.h5')
    folder_path = r"E:\Face Verification\senet-face\code\check_liveness\liveData"

    trues_1 = 0
    trues_2 = 0
    if os.path.isdir(folder_path):
        # Get all file names in the directory
        file_names = os.listdir(folder_path)
        for file in file_names:

            new_path = os.path.join(folder_path, file)
            print(new_path)
            d = True
            for video_file in os.listdir(new_path):

                if video_file.endswith('.wmv'):
                    video_path = os.path.join(new_path, video_file)
                    image_path = os.path.join(new_path, video_file[:-4])+".jpg"
                    faces, frames, boxes = get_video_frames_v1(video_path)
                    cv2.show()
                    image = image_path
                    res_verification = verify.pridect_with_frames(image, frames)
                    print("model Facenet (num 1) result: ")
                    # print(res_verification)
                    if len(res_verification['verified']) > 0:
                        d = max(res_verification['verified'],key=res_verification['verified'].count)
                        print(max(res_verification['verified'],key=res_verification['verified'].count))

                        if d == True:
                            trues_1 += 1

                    else:
                        print("False")

                    print("model Senet (num 2) result: ")
                    img = cv2.imread(image)
                    res_verification2 = verify_v2.predict_with_detect_face_frames(frames, img)
                    # print(res_verification2)
                    if len(res_verification2['verified']) > 0:
                        d = max(res_verification2['verified'],key=res_verification2['verified'].count)
                        print(max(res_verification2['verified'],key=res_verification2['verified'].count))

                        if d == True:
                            trues_2 += 1
                    else:
                        print("False")


            print(f"========{d}=============")

    print(trues_1, trues_2)









if __name__=="__main__":
    main()

