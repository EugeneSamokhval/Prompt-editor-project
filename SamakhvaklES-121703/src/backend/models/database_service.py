
import os
from contextlib import contextmanager
from typing import Iterable, List, Tuple

import torch
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    ForeignKey,
    select,
    BigInteger
)
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("VECTOR_DB_URL")
engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)
Base = declarative_base()

# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=False, nullable=False)
    encoded_password = Column(String, nullable=False)

    histories = relationship(
        "TestHistory", back_populates="user", cascade="all, delete-orphan"
    )


class TestHistory(Base):
    __tablename__ = "test_histories"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    image = Column(String, nullable=True)  # store path or URL to the image
    text = Column(Text, nullable=True)

    user = relationship("User", back_populates="histories")


class Document(Base):
    __tablename__ = "documents"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    source_id = Column(Text, nullable=False)
    embedding = Column(Vector(384), nullable=False, index=True)


# Create tables (make sure pgvector extension exists in DB)
Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# Sentence‑Transformers model & helpers
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
_model_name = "sentence-transformers/all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_model_name).to(device)
    return _model


def embed_texts(texts: Iterable[str]) -> List[List[float]]:
    """Encode strings into 384‑dim embeddings (lists of floats)."""
    embedder = get_embedder()
    return embedder.encode(list(texts), convert_to_numpy=True).tolist()


# ---------------------------------------------------------------------------
# Session helper
# ---------------------------------------------------------------------------

@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# CRUD & vector‑search functions
# ---------------------------------------------------------------------------

def save_user(email: str, username: str,  encoded_password: str) -> int:
    """Insert a new user and return its id."""
    with session_scope() as session:
        user = User(email=email, username=username, encoded_password=encoded_password)
        session.add(user)
        session.flush()  # assigns primary key
        return user.id


def save_test_history(
    user_id: int, *, image: str | None = None, text: str | None = None
) -> int:
    """Save a test history row and return its id."""
    with session_scope() as session:
        record = TestHistory(user_id=user_id, image=image, text=text)
        session.add(record)
        session.flush()
        return record.id

def search_similar(
    query_text: str,
    top_k: int = 5,
) -> List[Tuple[Document, float]]:
    """Return top‑k similar documents with (Document, distance) tuples."""
    query_emb = embed_texts([query_text])[0]
    distance_col = Document.embedding.cosine_distance(query_emb).label("distance")

    with session_scope() as session:
        distance_col = Document.embedding
        stmt = (
            select(Document, distance_col)
            .order_by(distance_col)
            .limit(top_k)
        )
        return session.execute(stmt).all()
