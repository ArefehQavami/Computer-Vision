import os
import cv2
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from imutils.video import FileVideoStream
from glob import glob
import base64

"""Files & Folders Section Methods"""


def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def build_path(root_path, post_path):
    return './{}/{}'.format(root_path, post_path)


def get_list_of_files(path, extention):
    return glob(path + extention)


"""Image Section Methods"""


def load_image_Image(image_path):
    return Image.open(image_path)


def load_image(image_path):
    return cv2.imread(image_path)


def convert_bgr2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def crop_image(image, box):
    return image[box[2]:box[3], box[0]:box[1]]


def save_image(path, image):
    cv2.imwrite(path, image)


def get_similarity(embeds):
    return cosine_similarity(embeds[0], embeds[1])


def resize_image(image, size=(64, 64)):
    return cv2.resize(image, size)


def get_video_frames_v2(video_path, video_len=10):
    v_cap = FileVideoStream(video_path).start()
    v_len = int(v_cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for j in range(v_len):
        frame = v_cap.read()
        frames.append(frame)

        if j > video_len:
            break
    return frames


def get_video_frames_v1(video_path, video_len=6):  # 50):
    capture = cv2.VideoCapture(video_path)
    counter = 0
    frames = []
    can_rad = False
    while capture.isOpened():
        can_rad = True
        ret, img = capture.read()
        if ret:
            if counter > video_len:
                img_det = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img_det)

            if counter > 41:
                break
            counter += 1
        else:
            break

    return frames


def save_file(path, content):
    open(path, 'wb').write(content)


def save_base65_file(path, data):
    try:
        content = base64.b64decode(data)
        open(path, 'wb').write(content)
    except Exception as e:
        print(e)


def get_model():
    return Face
