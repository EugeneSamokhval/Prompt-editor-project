import os
import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class LlamaTestService:
    def __init__(self):
        self._url = "https://api.segmind.com/v1/llama-v3p1-8b-instruct"
        
    async def test_prompt(self, prompt: str)->str:
        data = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            },
        ]
        }
        response = await requests.post(self._url, json=data, headers={'x-api-key': os.getenv('LLAMA_3_API_TOKEN')})
        return response.json()
