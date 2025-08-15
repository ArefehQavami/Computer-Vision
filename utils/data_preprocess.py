import glob
import cv2
import os
from mtcnn.mtcnn import MTCNN
import random
import time
import shutil


class DataPreprocess:
    """
    this a preprocess class
    we want to remove some unusable images and prepare others for next analysis
    such as:
    removing none type images
    removing very small images
    detecting faces from images
    """

    def __init__(self, config):
        self.config = config
        self.data_folder_path = self.config['data_folder_path']
        self.image_area = int(self.config['image_area'])
        self.videoFrame_folder_path = self.config['videoFrame_folder_path']
        self.min_face_area = self.config['min_face_area']

    def show_image(self, winname, img):
        '''
        Shows an image with the window name
        :param winname: name of shown window
        :param img: numpy image
        :return: window with image on it
        '''

        cv2.namedWindow(winname)
        cv2.moveWindow(winname, 40, 30)  # Move it to (40,30)
        cv2.imshow(winname, img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def get_video_frames(self):
        """
        creates frames from video
        and saves the even ones
        """
        types = ('/*.mp4', '/*.wmv', '/*.MOV')
        for folder_name in glob.glob(self.videoFrame_folder_path + '/*'):
            lst_img = []
            for files in types:
                lst_img.extend(glob.glob(folder_name + files))

            capture = cv2.VideoCapture(lst_img[0])
            counter = 1
            while capture.isOpened():
                ret, img = capture.read()
                if not ret:
                    break
                if counter < 10:
                    name = str(lst_img[0][-14:-4]) + '_00' + str(counter) + ".jpg"
                elif counter > 9 and counter < 100:
                    name = str(lst_img[0][-14:-4]) + '_0' + str(counter) + ".jpg"
                else:
                    name = str(lst_img[0][-14:-4]) + '_' + str(counter) + ".jpg"
                cv2.imwrite(folder_name + '\\' + name, img)
                counter += 1

    def remove_noneType_image(self):
        """
        if an image type is noneType and it can not be shown, delete it
        """
        none_type = type(None)
        for item in glob.glob(self.videoFrame_folder_path + '/*'):
            if os.path.isdir(item):
                for image in glob.glob(str(item) + '/*'):
                    img = cv2.imread(image)
                    if isinstance(img, none_type):
                        os.remove(str(image))
            else:
                image = item
                img = cv2.imread(image)
                if isinstance(img, none_type):
                    os.remove(str(image))

    def save_detect_image_face(self):
        """
        searchs in a folder
        detects faces
        and removes images not having face
        """
        none_type = type(None)
        for item in glob.glob(self.videoFrame_folder_path + '/*'):
            start_time_folder = time.time()
            if os.path.isdir(item):
                for image in glob.glob(str(item) + '/*'):
                    if image is not None:
                        face = self.detect_face(image)
                        if isinstance(face, none_type):
                            os.remove(str(image))
                        else:
                            cv2.imwrite(image, face)
            else:
                image = item
                face = self.detect_face(image)
                if isinstance(face, none_type):
                    os.remove(str(image))
                else:
                    cv2.imwrite(image, face)
            spent_time_folder = time.time() - start_time_folder
            print("time", spent_time_folder)

    def detect_face(self, image):
        """
        detects faces in an image and chooses the face with maximum area
        deletes other abnormal images that can not be detected
        :param image: numpy image
        :return: face image
        """
        detector = MTCNN()
        face_image = None
        pixels = cv2.imread(image)
        if pixels is None:
            return None
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
        if len(pixels.shape) == 3:
            a , b, c = pixels.shape
            # if image has three channels
            if c == 3:
                faces = detector.detect_faces(pixels)
                list_faces = []
                for face in faces:
                    x1, y1, width, height = face['box']
                    x2, y2 = x1 + width, y1 + height
                    face_image = pixels[y1:y2, x1:x2]
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                    (a, b, c) = face_image.shape
                    print(a+b)
                    if a + b > self.min_face_area:
                        list_faces.append(a+b)

                if len(list_faces) == 1:
                    index = list_faces.index(max(list_faces))
                    face = faces[index]
                    x1, y1, width, height = face['box']
                    x2, y2 = x1 + width, y1 + height
                    face_image = pixels[y1:y2, x1:x2]
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                else:
                    print("could not find any face")
                    return None

        return face_image

    def remove_small_image(self):
        """
        removes images smaller than area specified
        """
        for item in glob.glob(self.videoFrame_folder_path  + '/*'):
            if os.path.isdir(item):
                for image in glob.glob(str(item) + '/*'):
                    img = cv2.imread(image)
                    width, height, dimension = img.shape
                    if width + height < self.image_area:
                        os.remove(str(image))
            else:
                image = item
                img = cv2.imread(image)
                width, height, dimension = img.shape
                if width + height < self.image_area:
                    os.remove(str(image))

    def pick_random_items(self, num):
        """
        pick random items from a folder
        to the number of num
        :param num: number of items
        """
        for item in glob.glob(self.videoFrame_folder_path + '/*'):
            if os.path.isdir(item):
                list_img = []
                for image in glob.glob(str(item) + '/*'):
                    if image[-7:-4] != '000':
                        list_img.append(image)
                print(item)
                print(len(list_img))
                if len(list_img) <num :
                        continue
                num_to_select = num
                list_of_random_items = random.sample(list_img, num_to_select)
                lis_remove = [x for x in list_img if x not in list_of_random_items]
                for item in lis_remove:
                    os.remove(item)

    def add_000_to_image(self):
        """
        adds _000 to the name of first image in a folder
        """
        for item in glob.glob(self.videoFrame_folder_path + '/*'):
            if os.path.isdir(item):
                for image in glob.glob(str(item) + '/*'):
                    os.rename(image, image[:-4] + '_000' + '.jpg')
                    break

    def find_redundant_data(self):
        """
        Finds redundant people based on national code
        """
        list_img = []
        for item in glob.glob(self.videoFrame_folder_path + '/*'):
            if os.path.isdir(item):
                for image in glob.glob(str(item) + '/*'):
                    print(image[-18:-8])
                    list_img.append(image[-18:-8])

        temp = set([x for x in list_img if list_img.count(x) < 3])
        print(len(temp))
        print(temp)


    def delete_face(self):
        """
        Finds redundant people based on national code
        """
        for item in glob.glob(self.videoFrame_folder_path + '/*'):
            if os.path.isdir(item):
                for image in glob.glob(str(item) + '/*'):
                    # if image[-8:-4] == "ront":
                    #     try:
                    #         shutil.move(image, "E:/Face Verification/senet-face/data/Test_Live/MelliCard")
                    #     except Exception as e:
                    #         continue

                    if image[-8:-4] == "face":
                        os.remove(str(image))

