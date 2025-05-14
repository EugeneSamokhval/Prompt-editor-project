import json
import time
import os
from dotenv import load_dotenv
import requests
import base64, io

load_dotenv()

class FusionBrainAPI:

    def __init__(self):
        self.URL = 'https://api-key.fusionbrain.ai/'
        self.AUTH_HEADERS = {
            'X-Key': f"Key {os.getenv('FUSION_BRAIN_OPEN_KEY')}",
            'X-Secret': f"Secret {os.getenv('FUSION_BRAIN_SECRET_KEY')}",
        }

    def get_pipeline(self):
        response = requests.get(self.URL + 'key/api/v1/pipelines', headers=self.AUTH_HEADERS)
        data = response.json()
        return data[0]['id']

    def generate(self, prompt, pipeline, images=1, width=1024, height=1024):
        params = {
            "type": "GENERATE",
            "numImages": images,
            "width": width,
            "height": height,
            "generateParams": {
                "query": f'{prompt}'
            }
        }

        data = {
            'pipeline_id': (None, pipeline),
            'params': (None, json.dumps(params), 'application/json')
        }
        response = requests.post(self.URL + 'key/api/v1/pipeline/run', headers=self.AUTH_HEADERS, files=data)
        data = response.json()
        return data['uuid']

    def check_generation(self, request_id, attempts=10, delay=10)->bytes:
        while attempts > 0:
            response = requests.get(self.URL + 'key/api/v1/pipeline/status/' + request_id, headers=self.AUTH_HEADERS)
            data = response.json()
            if data['status'] == 'DONE':
                byte_data = base64.b64decode(data['result']['files'][0])
                return io.BytesIO(byte_data)

            attempts -= 1
            time.sleep(delay)
