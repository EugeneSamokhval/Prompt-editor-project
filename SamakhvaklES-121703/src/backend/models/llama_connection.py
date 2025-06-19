import os
import httpx
from dotenv import load_dotenv

load_dotenv()

class DeepSeekTestService:
    def __init__(self):
        self._url = "https://api.deepseek.com/chat/completions"
        self._api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self._api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
        
    async def improve_prompt(self, prompt: str):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._api_key}'
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a senior prompt engineer. You tasked with improving quality of every prompt you getting to the absolute best quality. Use every suitable technic to make prompt better. Don't say anything else except the result. Your answer must be plain text without markdown code"},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._url,
                    headers=headers,
                    json=payload,
                    timeout=120.0
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
                
        except httpx.HTTPStatusError as e:
            raise Exception(f"API request failed with status {e.response.status_code}")
        
    async def test_prompt(self, prompt: str) -> str:
        """Send prompt to DeepSeek API and return the response content"""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._api_key}'
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._url,
                    headers=headers,
                    json=payload,
                    timeout=120.0
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
                
        except httpx.HTTPStatusError as e:
            raise Exception(f"API request failed with status {e.response.status_code}")
