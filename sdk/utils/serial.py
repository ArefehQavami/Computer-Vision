import cv2
import yaml
from cardDetector.cardDetector import NationalCardDetector
from base.algorithm import Singleton


class Card(Singleton):
    def __init__(self):
        self.yaml_path = "./cardDetector/config.yaml"
        self.stream = open(self.yaml_path, 'r')
        self.config = yaml.load(self.stream)
        self.detecor = NationalCardDetector(self.config)


    def card_front_check(self, path):
        img = cv2.imread(path)
        try:
            cropped_img, res = self.detecor.frontTemplateMatching(img)
            if len(cropped_img) == 0:
                raise Exception('NOT FOUND !!!')
                return None
            return True

        except Exception as e:
            raise e

    def card_back_check(self, path):
        img = cv2.imread(path)
        try:
            cropped_img, res = self.detecor.backTemplateMatching(img)
            if len(cropped_img) == 0:
                raise Exception('NOT FOUND !!!')
                return None
            return True

        except Exception as e:
            raise e