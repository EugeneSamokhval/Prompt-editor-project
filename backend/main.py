import uvicorn
from models.prompts_tokenizer import PromptTokenizer
from models.prompt_rating import PromptRatingHandler, InputTypes
from models.FusionBrainConnection import FusionBrainAPI
from models.llama_connection import DeepSeekTestService
from models.database_service import *
# from models.database_helpers import *
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi import FastAPI

from dto import PromptStructure
app = FastAPI()

@app.get('/tokenize-prompt')
async def tokenize_prompt(prompt: str)->PromptStructure:
    prompt = prompt.replace('-', ' ')
    tokenizer = PromptTokenizer()
    structured_prompt = tokenizer.tokenize_prompt(prompt)
    return structured_prompt

@app.get('/rate-prompt')
async def rate_prompt(prompt: str, model: InputTypes):
    prompt = prompt.replace('-', ' ')
    rating_handler = PromptRatingHandler()
    rating = rating_handler.calculate(prompt=prompt, ai_type=model)
    return rating

@app.get('/preview-image')
async def get_image_preview(prompt: str):
    prompt = prompt.replace('-', ' ')
    image_gen_api = FusionBrainAPI()
    pipeline_id =  image_gen_api.get_pipeline()
    uuid = image_gen_api.generate(prompt, pipeline_id)
    files = image_gen_api.check_generation(uuid)
    while not files:
        files = image_gen_api.check_generation(uuid)
    return StreamingResponse(content=files, media_type='image/jpeg')
    
@app.get('/preview-text')
async def get_text_preview(prompt: str):
    prompt = prompt.replace('-', ' ')
    llama_service = DeepSeekTestService()
    response_message = await llama_service.test_prompt(prompt=prompt)
    return response_message

@app.get('/improve-prompt')
async def improve_prompt(prompt: str):
    prompt = prompt.replace('-', ' ')
    service = DeepSeekTestService()
    improved_prompt =  await service.improve_prompt(prompt)
    return improved_prompt

@app.get("/extend-image-prompt")
async def improve_image_prompt(prompt: str):
    prompt = prompt.replace('-', ' ')
    service = DeepSeekTestService()
    improved_prompt =  await service.improve_prompt(prompt)
    return improved_prompt

# @app.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
# def register(user_in: UserCreate):
#     """Create a new user account."""
#     if get_user_by_email(user_in.email):
#         raise HTTPException(status_code=400, detail="Email already registered")

#     hashed_pw = get_password_hash(user_in.password)
#     user_id = save_user(user_in.email, user_in.username, hashed_pw)

#     with session_scope() as session:
#         user = session.get(User, user_id)
#         return user


# @app.post("/token", response_model=Token)
# def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
#     """OAuth2 password flow – returns a JWT access token."""
#     # OAuth2PasswordRequestForm uses "username" field; we treat it as email.
#     user = authenticate_user(form_data.username, form_data.password)
#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect email or password",
#             headers={"WWW-Authenticate": "Bearer"},
#         )

#     access_token = create_access_token(data={"sub": str(user.id)})
#     return {"access_token": access_token, "token_type": "bearer"}


# @app.get("/history", response_model=List[HistoryItem])
# def read_history(current_user: User = Depends(get_current_user)):
#     """Return the current user's test history."""
#     with session_scope() as session:
#         histories = (
#             session.execute(
#                 select(TestHistory).where(TestHistory.user_id == current_user.id)
#             )
#             .scalars()
#             .all()
#         )
#         return histories
    

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