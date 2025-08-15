import json
import requests
from service.cdn import Cdn
from dotenv import load_dotenv
from server.settings import BASE_DIR
from base.algorithm import Singleton

load_dotenv(BASE_DIR / "server/.env")


class BPMS(Singleton):
    def __init__(self):
        self.url = ''

        self.payload = {}

        self.headers = {'Content-type': 'application/json'}
        self.cdn = Cdn()

    def task_status(self, p_id):
        try:
            url = f''

            response = requests.get(url, verify=False, headers=self.headers)
            response.raise_for_status()
            return response.status_code

        except:
            return response.status_code

    def task_complete(self, p_id, urn, res, base64_frame=''):
        for t in range(3):
            try:
                if not self.cdn.iam.Authorization:
                    self.cdn.set_auth()

                self.headers["Authorization"] = self.cdn.iam.Authorization
                url = f'{os.getenv("URL")}customer/definition/ai-submit-ekyc-inquiry-result'
                self.payload = {
                    "ekycSessionId": p_id,
                    "documentUrn": urn,
                    "verified": res,
                    "description": "",
                    "base64Frame": base64_frame
                }

                response = requests.put(url, data=json.dumps(self.payload), verify=False, headers=self.headers)
                response.raise_for_status()
                break

            except:
                print(f'{response.status_code}')

                if response.status_code == 401:
                    self.cdn.set_auth()

                if t == 2:
                    raise ValueError(f'{response.status_code}')
