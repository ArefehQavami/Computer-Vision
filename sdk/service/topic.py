import os
import json
import base64
import requests
from service.cdn import Cdn
from tqdm import tqdm
from service.ocr import CardOcr
from dotenv import load_dotenv
from server.settings import BASE_DIR
from process.models import Process
from base.algorithm import Singleton
from django.core.files.storage import default_storage
from commons.util import make_directory, save_file
from time import sleep
load_dotenv(BASE_DIR / "server/.env")


class Topic(Singleton):
    gpu_count = 0
    cpu_count = 0

    def __init__(self):
        self.headers = {'Content-type': 'application/json'}
        self.payload = {
            "batchSize": int(os.getenv("NUM"))
        }
        self.url = ''
        if int(os.getenv("ENV")) == 13:
            self.url = f'{os.getenv("URL")}customer/definition/ai-ekyc-document-inquiry?batchSize={int(os.getenv("NUM"))}'
        else:
            self.url = f'{os.getenv("URL")}customer/definition/ai-ekyc-document-inquiry'
        self.cdn = Cdn()
        # self.ocr = CardOcr()

    def fetch_data(self):
        try:
            if not self.cdn.iam.Authorization:
                self.cdn.refresh()

            self.headers["Authorization"] = self.cdn.iam.get_val()[0]
            processes = requests.get(self.url, data=json.dumps(self.payload), verify=False, headers=self.headers)
            processes.raise_for_status()

            processes = processes.json()
            for process in tqdm(processes):

                if Process.objects.filter(process_id=process['ekycSessionId'],
                                          process_type=Process.Type.AUTH).count() == 0:

                    _, _ = Process.objects.get_or_create(process_id=process['ekycSessionId'],
                                                         process_json=process,
                                                         process_status=Process.Status.INIT,
                                                         process_type=Process.Type.AUTH)
                else:
                    p = Process.objects.get(process_id=process['ekycSessionId'])
                    if (p.process_status != Process.Status.INIT) or (p.process_status != Process.Status.PENDING):

                        _, _ = Process.objects.get_or_create(process_id=process['ekycSessionId'],
                                                             process_json=process,
                                                             process_status=Process.Status.INIT,
                                                             process_type=Process.Type.AUTH)


        except Exception as e:
            if '401' in str(e):
                sleep(0.1)
                self.cdn.refresh()


            self.fetch_data()

    def obj(self, o):
        if isinstance(o, dict):
            return o
        elif isinstance(o, str):
            return json.loads(o)
        else:
            return json.loads(json.dumps(o))

    def preprocess(self, process):
        dest_path = default_storage.path(process.process_id)
        make_directory(dest_path)

        nin = process.process_json['nationalCode']
        v_urn = process.process_json['ekycDocumentPackage']['ekycVideoUrn']
        img_urn = process.process_json['ekycDocumentPackage']['ncrImageUrn']
        is_pass = True

        process_url = {}  # if process.process_url == None else json.loads(json.dumps(process.process_url))
        process_err = {} if process.process_errors == None else self.obj(process.process_errors)

        # Video
        if v_urn != None:
            try:
                process_url["video"] = f'{default_storage.url(process.process_id)}/{nin}.wmv'
                self.save_file(f"{dest_path}/{nin}.wmv", v_urn)
            except Exception as e:
                process_err[v_urn] = str(e)
                is_pass = False
        else:
            is_pass = False

        # image
        if img_urn != None:
            try:
                process_url["image"] = f'{default_storage.url(process.process_id)}/{nin}.jpg'
                self.save_file(f"{dest_path}/{nin}.jpg", img_urn)
            except Exception as e:
                process_err[img_urn] = str(e)
                is_pass = False
        else:
            is_pass = False

        if not is_pass:
            process.process_status = Process.Status.ERROR

            try:
                self.cdn.task_complete_auth(p_id=process.process_id, urn=v_urn, res=False, desc="")

            except:
                process.process_try = 3

        process.process_url = json.dumps(process_url)
        process.process_errors = process_err
        process.save()

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


import pandas as pd


class Count(Singleton):
    def __init__(self):
        self.df = pd.DataFrame([0, 0])
        self.df.to_csv('config/df.csv')

    def update_a(self, key):
        self.df = pd.read_csv('config/df.csv', index_col=0)
        self.df.iloc[key, 0] += 1
        self.df.to_csv('config/df.csv')

    def update_m(self, key):
        self.df = pd.read_csv('config/df.csv', index_col=0)
        self.df.iloc[key, 0] -= 1
        self.df.to_csv('config/df.csv')

    def get(self):
        self.df = pd.read_csv('config/df.csv', index_col=0)
        return f'gc {self.df.iloc[0, 0]} - cc {self.df.iloc[1, 0]}'

    def get_val(self, key):
        self.df = pd.read_csv('config/df.csv', index_col=0)
        return self.df.iloc[0, key]
