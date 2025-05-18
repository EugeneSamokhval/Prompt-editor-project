import uvicorn
from models.prompts_tokenizer import PromptTokenizer
from models.prompt_rating import PromptRatingHandler, InputTypes
from models.FusionBrainConnection import FusionBrainAPI
from models.llama_connection import LlamaTestService
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi import FastAPI

from dto import PromptStructure
app = FastAPI()

@app.get('/tokenize-prompt')
async def tokenize_prompt(prompt: str)->PromptStructure:
    tokenizer = PromptTokenizer()
    structured_prompt = tokenizer.tokenize_prompt(prompt)
    return structured_prompt

@app.get('/rate-prompt')
async def rate_prompt(prompt: str, model: InputTypes):
    rating_handler = PromptRatingHandler()
    rating = rating_handler.calculate(prompt=prompt, ai_type=model)
    return rating

@app.get('/preview-image')
async def get_image_preview(prompt: str):
    image_gen_api = FusionBrainAPI()
    pipeline_id =  image_gen_api.get_pipeline()
    uuid = image_gen_api.generate(prompt, pipeline_id)
    files = image_gen_api.check_generation(uuid)
    while not files:
        files = image_gen_api.check_generation(uuid)
    return StreamingResponse(content=files, media_type='image/jpeg')
    
@app.get('/preview-text')
async def get_text_preview(prompt: str):
    llama_service = LlamaTestService()
    response_message = await llama_service.test_prompt(prompt=prompt)
    return response_message

origins = [
   'http://localhost:8000',
   'http://localhost:3000',
   'http://localhost:5173'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["traceparent", "filename"]
)

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, log_level="debug", reload=True)