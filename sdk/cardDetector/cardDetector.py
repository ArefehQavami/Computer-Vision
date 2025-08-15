import cv2
import numpy as np
import math
from itertools import combinations
import sys
import random as rng
import glob
import imutils
from mtcnn.mtcnn import MTCNN
from termcolor import colored


class NationalCardDetector:
    '''
    This is a detector class for Iran National Card in order to crop the card and check it's completeness
    It works based on template matching and finding rectangle contours
    To apply these methods first the image is preprocessed
    It has to be mentioned that before finding template matching functional we used corner detection
    '''

    def __init__(self, config):
        self.config = config
        self.contour_param1 = cv2.RETR_TREE
        self.contour_param2 = cv2.CHAIN_APPROX_NONE
        self.threshold_type = cv2.THRESH_BINARY_INV
        self.canny_high_thresh = self.config['canny_high_thresh']
        self.canny_low_thresh = self.config['canny_low_thresh']
        self.front_template, self.front_tH, self.front_tW = self.read_template_image(config['front_template_path'], 'Front')
        self.back_template, self.back_tH, self.back_tW = self.read_template_image(config['back_template_path'], 'Back')
        self.min_num_region = int(self.config['minimum_number_region'])
        self.height = self.config['img_resize_height']
        self.width = self.config['img_resize_width']
        self.region_avg_thresh = self.config['region_avg_threshold']


    def read_image(self, image_path):
        '''
        Reads an image with cv2
        :param image_path: string path of an image
        :return: image in numpy format
        '''

        img = cv2.imread(image_path)

        return img

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

    def resize_image_byScale(self, img, scale):
        '''
        Resizes image to a defined shape
        returns the resized image
        :param img: numpy image
        :param scale: number which image is multiply to
        :return: resized image
        '''

        img_resized = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), 0, 0, interpolation=cv2.INTER_AREA)

        return img_resized

    def resize_image_byShape(self, img, width, height):
        '''
        Resizes image to a defined shape
        returns the resized image
        :param img: numpy image
        :param width: width of image
        :param height: height of image
        :return: resized image
        '''

        img_resized = cv2.resize(img, (width, height))

        return img_resized


    def gray_image(self, img):
        '''
        Turns image to gray scale
        returns the gray image
        :param img: numpy image
        :return: grayScale of image
        '''

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img_gray


    def blur_image(self, img_gray, kernel_size):
        '''
        Blurs an image with median or gaussian filter
        returns the blurred image
        :param img_gray: image in grayscale
        :param kernel_size: blur kernel size
        :return: blurred image
        '''

        img_blur = cv2.medianBlur(img_gray, kernel_size)
        # img_blur= cv2.GaussianBlur(img_gray,(kernel_size, kernel_size),0)

        return img_blur

    def detect_edge_canny(self, image, high_thresh, low_thresh):
        '''
        Detects edges based on canny edge detector
        the lower and upper thresholds are manual and given by user
        :param image: numpy image
        :param high_thresh: high threshold value of intensity gradient
        :param low_thresh: low threshold value of intensity gradient
        :return: image of edges
        '''

        img_edges = cv2.Canny(image, high_thresh, low_thresh, apertureSize=3)

        return img_edges

    def list_folder_images(self, folderPath):
        '''
        Finds all images in a folder
        returns a list of image paths
        :param folderPath: path of folder with images in it
        :return: list of images names
        '''

        types = ('/*.jpg', '/*.jpeg', '/*.png')
        lst_img = []
        for files in types:
            lst_img.extend(glob.glob(folderPath + files))

        return lst_img

    def read_template_image(self, tmp_path, choose):
        '''
        Reads template image and gray scale it
        apply canny to detect edges
        and find the template shape
        :param tmp_path: template path
        :param choose: side of National Card: Front or Back
        :return: numpy template, height of template, width of template
        '''

        template = self.read_image(tmp_path)
        template = self.gray_image(template)
        if choose == 'Back':
            template = self.blur_image(template, 3)
        template = self.detect_edge_canny(template, self.canny_high_thresh, self.canny_low_thresh)
        (tH, tW) = template.shape[:2]

        return template, tH, tW

    def frontTemplateMatching(self, img):
        '''
        Having a template of National Card's Front
        checks if the input image has a template in it
        and if yes, crops the image
        and checks if the cropped image has a face and some rectangle contours for items
        and checks if information on the front card has reflection of light or not
        :param img: numpy image
        :return: cropped image, True or None, False
        '''

        img_resized = self.resize_image_byShape(img, self.width, self.height)
        cropped_img, scale = self.template_matching(img_resized, 'Front')
        if cropped_img is not None:
            num_regions, rect = self.find_region(cropped_img)
            if num_regions > self.min_num_region and rect is not None:
                cropped_face_image = self.detect_face(cropped_img)
                if cropped_face_image is not None:
                    if self.check_light_front(cropped_img) is not None:
                        return cropped_img, True
                    else:
                        print("Light reflection on card")
                        raise Exception("Light reflection on card")
                        return None, False
                else:
                    print("could not find any face in image ")
                    raise Exception("could not find any face in image ")
                    return None, False
            else:
                print(num_regions, ", not enough contours")
                raise Exception(num_regions, ", not enough contours")

                return None, False
        else:
            print("could not find any template")
            raise Exception("could not find any template")
            return None, False

    def backTemplateMatching(self, img):
        '''
        Having a template of National Card's Back
        checks if the input image has a template in it
        and if yes, crops the image
        and checks if the cropped image has some rectangle contours for items
        and checks if barcode on the back card has reflection of light or not
        :param img: numpy image
        :return: cropped image, True-False or None, False
        '''

        img_resized = self.resize_image_byShape(img, self.width, self.height)
        cropped_img, scale = self.template_matching(img_resized, 'Back')
        if cropped_img is not None:
            num_regions, rect = self.find_region(cropped_img)
            if num_regions > self.min_num_region and rect is not None:
                if self.check_light_barcode(cropped_img) > int(self.config['minimum_number_barcode']):
                    return cropped_img, True
                else:
                    print("Light reflection on barcode")
                    return cropped_img, False
            else:
                print(num_regions, ", not enough contours")
                raise Exception(f'{num_regions} not enough contours')
                return [], False
        else:
            print("could not find any template")
            raise Exception("could not find any template")
            return [], False

    def template_matching(self, img, method_str):
        '''
        Takes a template and an image
        and checks if the template(rectangle) is in the image
        loops over the scales of the image
        resizes the image according to the scale, and keeps track of the ratio of the resizing
        if the resized image is smaller than the template, then breaks from the loop
        detects edges in the resized, grayscale image and applies template matching to find the template in the image
        draws a bounding box around the detected result and display the image
        then returns the cropped rectangle
        :param img: numpy image
        :param method_str: side of image, Front or Back
        :return: numpy cropped image or None
        '''

        img_gray = self.gray_image(img)
        if method_str == 'Front':
            template = self.front_template
            tH = self.front_tH
            tW = self.front_tW

        elif method_str == 'Back':
            template = self.back_template
            tH = self.back_tH
            tW = self.back_tW

        img_gray = self.blur_image(img_gray, 3)
        found = None
        sc = 1
        for scale in np.linspace(0.01, 1.0, 50):
            resized = imutils.resize(img_gray, width=int(img_gray.shape[1] * scale))
            r = img_gray.shape[1] / float(resized.shape[1])
            if resized.shape[0] < tH or resized.shape[1] < tW:
                continue
            edged = self.detect_edge_canny(resized, self.canny_high_thresh, self.canny_low_thresh)
            result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            if found is None or maxVal > found[0]:
                sc = scale
                found = (maxVal, maxLoc, r)

        if found is not None:
            (maxVal, maxLoc, r) = found
            (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
            (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
            crop_img = self.add_margin(img, startX, startY, endX, endY)
            if maxVal < float(self.config['template_matching_maxVal']):
                return None, None
            return crop_img, sc

        else:
            return None, None


    def add_margin(self, img, startX, startY, endX, endY):
        '''
        :param img: numpy cropped image
        :param startX: start x coordinates
        :param startY: start Y coordinates
        :param endX: end x coordinates
        :param endY: end y coordinates
        :return: cropped image with margin
        '''

        crop_margin = self.config["card_margin"]
        if startY - crop_margin < 0:
            startY = 0
        else:
            startY = startY - crop_margin
        if startX - crop_margin < 0:
            startX = 0
        else:
            startX = startX - crop_margin

        if endY + crop_margin > img.shape[0]:
            endY = endY
        else:
            endY = endY + crop_margin

        if endX + crop_margin > img.shape[1]:
            endX = endX
        else:
            endX = endX + crop_margin

        crop_img = img[startY:endY, startX:endX]

        return crop_img

    def detect_face(self, img):
        '''
        Detects faces in an image based on Vilo Jones algorithm
        and chooses the face with maximum area
        returns the detected face
        :param image: numpy cropped image
        :return: face of person or None
        '''

        detector = MTCNN()
        face_image = None
        pixels = img

        if len(pixels.shape) == 3:
            a, b, c = pixels.shape
            # if image has three channels
            if c == 3:
                faces = detector.detect_faces(pixels)
                if len(faces) != 0 and faces[0]['confidence'] > float(self.config["face_confidence_score"]):
                    list_faces = []
                    for face in faces:
                        x1, y1, width, height = face['box']
                        x2, y2 = x1 + width, y1 + height
                        face_image = pixels[y1:y2, x1:x2]
                        (a, b, c) = face_image.shape
                        list_faces.append(a + b)

                        if len(list_faces) > 0:
                            index = list_faces.index(max(list_faces))
                            face = faces[index]
                            x1, y1, width, height = face['box']
                            x2, y2 = x1 + width, y1 + height
                            face_image = pixels[y1:y2, x1:x2]
                else:
                    return None

        return face_image

    def find_region(self, img):
        '''
        Finds text and face regions in image based on contours found in shape of rectangle
        performs Binary Inverse threshold and specifies structure shape and kernel size
        kernel size increases or decreases the area of the rectangle to be detected.
        a smaller value like (3, 3) will detect each word instead of a sentence
        looping through the identified contours then rectangular part is drawn
        returns an image with rectangles on it and num of regions found
        :param img: numpy cropped image
        :return: number of regions
        '''

        img_gray = self.gray_image(img)
        avg = np.average(img_gray)
        blur_img = self.blur_image(img_gray, 3)
        thresh = avg - self.region_avg_thresh
        if avg - self.region_avg_thresh < 0:
            thresh = 0
        ret, thresh1 = cv2.threshold(blur_img, thresh, 255, self.threshold_type)
        kernel_size = self.config['region_kernel_size']
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, rect_kernel)
        contours, hierarchy = cv2.findContours(closing, self.contour_param1, self.contour_param2)
        im2 = img.copy()
        big_contours = 0
        rect = None
        if len(contours) > 0:
            for cnt in contours:
                if cnt.shape[0] > int(self.config["barcode_shape_min"]):
                    big_contours = big_contours + 1
                    x, y, w, h = cv2.boundingRect(cnt)
                    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

            num_regions = big_contours
            return num_regions, rect
        else:
            return 0, None

    def check_light_front(self, img):
        '''
        Checks the reflection of light on front of national card
        in a way that if the information part of card has light on it
        it calculates the average brightness of that place
        if the area is too bright then the method rejects the card
        :param img: cropped numpy image
        :return: True or None
        '''

        img_gray = self.gray_image(img)
        avg = np.average(img_gray)
        y = img_gray.shape[0]
        x = img_gray.shape[1]
        blur_img = self.blur_image(img_gray, 3)
        thresh = avg - self.region_avg_thresh
        if avg - self.region_avg_thresh < 0:
            thresh = 0
        ret, thresh1 = cv2.threshold(blur_img, thresh, 255, self.threshold_type)
        info_part_thresh1 = thresh1[int(2 * y / 10):int(9 * y / 10), int(5 * x / 10):int(8 * x / 10)]
        thresh_avg = np.average(info_part_thresh1)
        # print("thresh", thresh_avg)
        if thresh_avg < float(self.config["threshold_front_light"]):
            print("image has light reflection")
            raise Exception("image has light reflection")
            return None
        else:
            return True

    def check_light_barcode(self, img):
        '''
        Checks the reflection of light on back of national card
        in a way that if the barcode part of card has light on it
        it calculates the average brightness of that place
        and calculates the number of rectangle contours in barcode area
        if the area is too bright or the number of contours is few
        then the method rejects the card
        :param img: cropped numpy image
        :return: number of contours
        '''

        img_resized = self.resize_image_byScale(img, 2)
        img_gray = self.gray_image(img_resized)
        y = img_gray.shape[0]
        x = img_gray.shape[1]
        barcode_part = img_gray[int(12.5 * y / 15):int(15 * y / 15), int(5.2 * x / 15):int(9.8 * x / 15)]
        light_avg = np.average(barcode_part)
        if light_avg > self.config['barcode_light_avg_threshold']:
            return 0
        high_thresh = self.config['barcode_canny_high_thresh']
        low_thresh = self.config['barcode_canny_low_thresh']
        thresh1 = self.detect_edge_canny(barcode_part, high_thresh, low_thresh)
        kernel_size = self.config['barcode_kernel_size']
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
        contours, hierarchy = cv2.findContours(dilation, self.contour_param1, self.contour_param2)
        rect = None
        barcode_cnt = 0
        if len(contours) > 0:
            for cnt in contours:
                if cnt.shape[0] > int(self.config["barcode_shape_min"]) and cnt.shape[0] < int(self.config["barcode_shape_max"]):
                    x, y, w, h = cv2.boundingRect(cnt)
                    rect = cv2.rectangle(barcode_part, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    barcode_cnt = barcode_cnt + 1

        return barcode_cnt


