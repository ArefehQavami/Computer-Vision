
import os
import numpy as np
import cv2
import base64
from pathlib import Path
from PIL import Image
import requests

from detectors.FaceDetector import Detector as FaceDetector

import tensorflow as tf

tf_version = tf.__version__
tf_major_version = int(tf_version.split(".")[0])
tf_minor_version = int(tf_version.split(".")[1])

if tf_major_version == 1:
    import keras
    from keras.preprocessing.image import load_img, save_img, img_to_array
    from keras.applications.imagenet_utils import preprocess_input
    from keras.preprocessing import image
elif tf_major_version == 2:
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
    from tensorflow.keras.applications.imagenet_utils import preprocess_input
    from tensorflow.keras.preprocessing import image


# --------------------------------------------------

def initialize_input(img1_path, img2_path=None):
    if type(img1_path) == list:
        bulkProcess = True
        img_list = img1_path.copy()
    else:
        bulkProcess = False

        if (
                (type(img2_path) == str and img2_path != None)  # exact image path, base64 image
                or (isinstance(img2_path, np.ndarray) and img2_path.any())  # numpy array
        ):
            img_list = [[img1_path, img2_path]]
        else:
            img_list = [img1_path]

    return img_list, bulkProcess


def initialize_folder():
    home = get_home()

    if not os.path.exists(home + "/.deep"):
        os.makedirs(home + "/.deep")
        print("Directory created")

    if not os.path.exists(home + "/.deep/weights"):
        os.makedirs(home + "/.deep/weights")
        print("Directory created")


def get_home():
    return str(os.getenv('EKYC_HOME', default=Path.home()))


def loadBase64Img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def load_image(img):
    exact_image = False;
    base64_img = False;
    url_img = False

    if type(img).__module__ == np.__name__:
        exact_image = True

    elif len(img) > 11 and img[0:11] == "data:image/":
        base64_img = True

    elif len(img) > 11 and img.startswith("http"):
        url_img = True

    # ---------------------------

    if base64_img == True:
        img = loadBase64Img(img)

    elif url_img:
        img = np.array(Image.open(requests.get(img, stream=True).raw).convert('RGB'))

    elif exact_image != True:  # image path passed as input
        if os.path.isfile(img) != True:
            raise ValueError("Confirm that ", img, " exists")

        img = cv2.imread(img)

    return img


def detect_face(img, detector_backend='opencv', face_detector=None, grayscale=False, enforce_detection=True,
                align=True):

    img_region = [0, 0, img.shape[0], img.shape[1]]

    # ----------------------------------------------

    if detector_backend == 'skip':
        return img, img_region

    # ----------------------------------------------

    detector = FaceDetector()

    try:
        detected_face, img_region = detector.detect_face(face_detector=face_detector,
                                                             detector_backend=detector_backend,
                                                             img=img,
                                                             align=align)

    except Exception as e:
        print(e)
        detected_face = None

    if (isinstance(detected_face, np.ndarray)):
        return detected_face, img_region
    else:
        if detected_face == None:
            if enforce_detection != True:
                return img, img_region
            else:
                raise ValueError(
                    "Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False.")


def normalize_input(img, normalization='base'):


    if normalization == 'base':
        return img
    else:


        img *= 255

        if normalization == 'raw':
            pass

    return img


def preprocess_face(img, target_size=(224, 224), grayscale=False, enforce_detection=True, detector_backend='opencv',
                    face_detector=None, return_region=False, align=True, is_face=False):


    if not is_face:
        img = load_image(img)
    base_img = img.copy()

    img, region = detect_face(img=img
                              ,detector_backend=detector_backend
                              ,face_detector=face_detector
                              ,grayscale=grayscale
                              ,enforce_detection=enforce_detection
                              ,align=align)

    # --------------------------

    if img.shape[0] == 0 or img.shape[1] == 0:
        if enforce_detection == True:
            raise ValueError("Detected face shape is ", img.shape,
                             ". Consider to set enforce_detection argument to False.")
        else:  # restore base image
            img = base_img.copy()

    # --------------------------


    if grayscale == True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------

    if img.shape[0] > 0 and img.shape[1] > 0:
        factor_0 = target_size[0] / img.shape[0]
        factor_1 = target_size[1] / img.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
        img = cv2.resize(img, dsize)

        # Then pad the other side to the target size by adding black pixels
        diff_0 = target_size[0] - img.shape[0]
        diff_1 = target_size[1] - img.shape[1]
        if grayscale == False:
            # Put the base image in the middle of the padded image
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)),
                         'constant')
        else:
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

    # ------------------------------------------


    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    # ---------------------------------------------------


    img_pixels = image.img_to_array(img)  # what this line doing? must?
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255  # normalize input in [0, 1]

    # ---------------------------------------------------

    if return_region == True:
        return img_pixels, region
    else:
        return img_pixels


def find_input_shape(model):
    # face recognition models have different size of inputs

    input_shape = model.layers[0].input_shape

    if type(input_shape) == list:
        input_shape = input_shape[0][1:3]
    else:
        input_shape = input_shape[1:3]

    # ----------------------

    if tf_major_version == 2 and tf_minor_version >= 5:
        x = input_shape[0];
        y = input_shape[1]
        input_shape = (y, x)

    # ----------------------

    if type(input_shape) == list:
        input_shape = tuple(input_shape)

    return input_shape

def create_resp(res, step=True):
    if step == True:
        res = (1 - np.average(res['distance'])) * 100
        return float(f"{res:.2f}")
    else:
        res = np.average(res['distance']) * 100
        return float(f"{res:.2f}")
