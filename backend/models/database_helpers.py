# import os
# from datetime import datetime, timedelta
# from typing import List, Optional

# from fastapi import FastAPI, Depends, HTTPException, status
# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# from pydantic import BaseModel, EmailStr
# from jose import JWTError
# from passlib.context import CryptContext
# from sqlalchemy import select

# # -----------------------------------------------------------------------------
# # Import your DB helpers & models
# # -----------------------------------------------------------------------------
# # Replace "service" with the module name that contains your earlier code.
# from database_service import User, TestHistory, session_scope, save_user

# # -----------------------------------------------------------------------------
# # Configuration (keep secrets out of source controlj!)
# # -----------------------------------------------------------------------------
# SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
# ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = 60

# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# app = FastAPI(title="Auth & History API")

# # -----------------------------------------------------------------------------
# # Pydantic Schemas – request / response bodies
# # -----------------------------------------------------------------------------
# class UserCreate(BaseModel):
#     email: EmailStr
#     username: str
#     password: str


# class UserOut(BaseModel):
#     id: int
#     email: EmailStr
#     username: str

#     class Config:
#         orm_mode = True


# class Token(BaseModel):
#     access_token: str
#     token_type: str


# class HistoryItem(BaseModel):
#     id: int
#     image: Optional[str]
#     text: Optional[str]

#     class Config:
#         orm_mode = True


# # -----------------------------------------------------------------------------
# # Helper functions (hash, JWT, user lookup / auth)
# # -----------------------------------------------------------------------------

# def verify_password(plain_password: str, hashed_password: str) -> bool:
#     return pwd_context.verify(plain_password, hashed_password)


# def get_password_hash(password: str) -> str:
#     return pwd_context.hash(password)


# def create_access_token(data: dict, expires_delta: int | None = None) -> str:
#     to_encode = data.copy()
#     expire = datetime.utcnow() + timedelta(
#         minutes=expires_delta or ACCESS_TOKEN_EXPIRE_MINUTES
#     )
#     to_encode.update({"exp": expire})
#     return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# def get_user_by_email(email: str) -> Optional[User]:
#     """Return a User by e‑mail address (or None)."""
#     with session_scope() as session:
#         return (
#             session.execute(select(User).where(User.email == email))
#             .scalars()
#             .first()
#         )


# def authenticate_user(email: str, password: str) -> Optional[User]:
#     user = get_user_by_email(email)
#     if not user or not verify_password(password, user.encoded_password):
#         return None
#     return user


# async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         user_id: str | None = payload.get("sub")
#         if user_id is None:
#             raise credentials_exception
#     except JWTError:
#         raise credentials_exception

#     with session_scope() as session:
#         user = session.get(User, int(user_id))
#         if user is None:
#             raise credentials_exception
#         return user