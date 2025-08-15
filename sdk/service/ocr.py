import json
import base64
from utils.serial import Card
from process.models import Process
from base.algorithm import Singleton
from backoffice.models import Backoffice
from commons.util import make_directory, save_file
import tensorflow as tf
import os

class CardOcr(Singleton):
    def __init__(self, cdn):
        self.ocr = ''
        if os.getenv("OCR") == '1':
            self.ocr = Card()
        self.cdn = cdn

    def set_detail(self, backoffice, backoffice_res):
        backoffice.process_res = json.dumps(backoffice_res)
        backoffice.save()

    def get_detail(self, backoffice):
        return {} if backoffice.process_res == None else json.loads(backoffice.process_res)

    def front_card(self, process, url):
        try:
            pk = process.process_id
            backoffice = Backoffice.objects.filter(process=process)[0]
            backoffice.process_status = Backoffice.Status.PENDING
            backoffice.save()


            backoffice_res = self.get_detail(backoffice)
            res_ocr = None


        except Exception as e:
            raise e

        try:
            res_ocr = self.ocr.card_front_check(url)
            backoffice.process_status = Backoffice.Status.COMPLETED
            backoffice_res['ocr_res_front'] = json.dumps(res_ocr)
            backoffice_res['ocr_status_front'] = True
            backoffice.save()

        except Exception as e:
            backoffice.process_status = Process.Status.FAILED
            backoffice.save()
            raise e

        return res_ocr

    def back_card(self, process, url):
        try:
            pk = process.process_id
            backoffice, _ = Backoffice.objects.get_or_create(process=process, process_type=Backoffice.Type.OCR,
                                                             process_status=Backoffice.Status.INIT)
            backoffice.process_status = Backoffice.Status.PENDING
            print(backoffice.process_status)
            backoffice.save()


            backoffice_res = self.get_detail(backoffice)
            res_ocr = None


        except Exception as e:
            raise e

        try:
            res_ocr = self.ocr.card_serial_code(url)
            backoffice.process_status = Backoffice.Status.COMPLETED
            backoffice_res['ocr_res_back'] = json.dumps(res_ocr)
            backoffice_res['ocr_status_back'] = True
            backoffice.save()

        except Exception as e:
            backoffice.process_status = Process.Status.FAILED
            backoffice.save()
            raise e

        return res_ocr

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
                print(e)
                raise Exception(e)
        else:
            raise Exception('urn is Null!!!')



