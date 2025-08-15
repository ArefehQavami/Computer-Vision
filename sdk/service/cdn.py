import datetime
import os
import json
import random

import requests
from dotenv import load_dotenv
from server.settings import BASE_DIR
from base.algorithm import Singleton
from time import sleep

load_dotenv(BASE_DIR / "server/.env")


class Cdn(Singleton):
    status = True

    def __init__(self):
        self.iam = Iam()
        self.headers = {'accept': '*/*', f"Authorization": self.iam.get_val()[0]}
        self.url = f'{os.getenv("URL")}cdn'

    def set_auth(self):
        try:
            self.iam.update_(False)
            if self.iam.pass_time():
                token = self.iam.login()
                self.headers = {'accept': '*/*', f"Authorization": token}

            else:
                self.headers = {'accept': '*/*', f"Authorization": self.iam.get_val()[0]}
            self.iam.update_(True)
        except Exception as e:
            print(e)
            self.iam.update_(True)
            raise e

    def fetch_from_cdn(self, urn):
        self.headers = {'accept': '*/*', f"Authorization": self.iam.get_val()[0]}
        response = requests.get(f'{self.url}/{urn}', headers=self.headers, verify=False)
        return response

    def refresh(self):

        if self.iam.get_val()[1] == 0:
            self.set_auth()
        else:
            while self.iam.get_val()[1]:
                self.iam.Authorization = self.iam.get_val()[0]
                sleep(0.25)

    def download_file_from_cdn(self, urn):

        try:
            # sleep(rnd)
            if not self.iam.Authorization:
                self.refresh()

            response = self.fetch_from_cdn(urn)

            if response.status_code != 200:
                response.raise_for_status()

        except:

            if response.status_code == 401:
                self.refresh()
            response = self.fetch_from_cdn(urn)

        return response.content, response.status_code

    def task_complete_auth(self, p_id, urn, res, base64_frame='', score=0, desc=""):
        cnt = 10
        for t in range(cnt):
            try:
                if not self.iam.Authorization:
                    self.refresh()

                headers = {
                    f"Authorization": self.iam.get_val()[0],
                    "Content-Type": 'application/json'
                }
                url = f'{os.getenv("URL")}customer/definition/ai-submit-ekyc-inquiry-result'

                self.payload = {
                    "ekycSessionId": p_id,
                    "documentUrn": urn,
                    "verified": res,
                    "base64Frame": base64_frame
                }

                if os.getenv("LOGGING_S") == '1':
                    self.payload["accommodationPercentage"] = score

                if os.getenv("LOGGING_D") == '1':
                    self.payload["description"] = desc

                response = requests.put(url, data=json.dumps(self.payload), verify=False, headers=headers)


                if response.status_code != 200 and response.status_code != 400:
                    response.raise_for_status()


                break

            except:
                sleep(0.5)

                if response.status_code == 401:
                    self.refresh()

                if response.status_code == 400:
                    raise ValueError(f'{response.status_code}')

                if t == (cnt - 1):
                    raise ValueError(f'{response.status_code}')

    def task_complete_ocr(self, p_id, urn, res, serial=''):
        for t in range(3):
            try:
                if not self.iam.Authorization:
                    self.refresh()
                headers = {
                    f"Authorization": self.iam.get_val()[0],
                    "Content-Type": 'application/json'
                }
                url = f'{os.getenv("URL")}customer/definition/ai-submit-ekyc-inquiry-result'

                self.payload = {
                    "ekycSessionId": p_id,
                    "documentUrn": urn,
                    "verified": res,
                    "description": "",
                    "nationalCardSerial": serial
                }
                response = requests.put(url, data=json.dumps(self.payload), verify=False, headers=headers)

                if response.status_code != 200:
                    response.raise_for_status()

                break

            except:

                if response.status_code == 401:
                    self.refresh()

                if t == 2:
                    raise ValueError(f'response-status = {response.status_code}')


import pandas as pd


class Iam(Singleton):
    Authorization = ''

    def __init__(self):
        self.name = 'sess.csv'
        self.url = f'{os.getenv("URL")}iam/auth/login'
        self.headers = {'Content-Type': 'application/json'}
        self.payload = {}
        if os.getenv("STAGE") == '0':
            self.payload = {
                "username": "ai.service",
                "password": "123456789"
            }
        else:
            self.payload = {
                "username": "ai.service",
                "password": "123456@Aa"
            }
        self.df = pd.DataFrame(['0', 0,  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        self.df.to_csv('config/sess.csv')
        _ = self.login()

    def login(self):
        r = requests.post(self.url, data=json.dumps(self.payload), verify=False, headers=self.headers)
        Iam.Authorization = r.headers['Authorization']
        self.update_t(r.headers['Authorization'])
        return r.headers['Authorization']

    def update_(self, is_done):
        self.df = pd.read_csv('config/sess.csv', index_col=0)
        self.df.iloc[1, 0] = 0 if is_done else 1
        self.df.to_csv('config/sess.csv')

    def update_t(self, t):
        self.df = pd.read_csv('config/sess.csv', index_col=0)
        self.df.iloc[0, 0] = t
        self.df.iloc[2, 0] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.df.to_csv('config/sess.csv')

    def pass_time(self):
        self.df = pd.read_csv('config/sess.csv', index_col=0)
        d = datetime.datetime.now() - datetime.datetime.strptime(self.df.iloc[2, 0], "%Y-%m-%d %H:%M:%S")
        if d.seconds // 60 > 30:
            return True
        else:
            return False

    def get_val(self):
        self.df = pd.read_csv('config/sess.csv', index_col=0)
        return self.df.iloc[0, 0], int(self.df.iloc[1, 0])
#