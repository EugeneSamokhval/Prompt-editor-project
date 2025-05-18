from pathlib import Path    
import pickle
import sys
import time

import numpy as np
from tqdm import tqdm

from sqlalchemy import BigInteger, Column, Text, create_engine, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import DeclarativeBase, Session
from pgvector.sqlalchemy import Vector

# ----------------------------------------------------------------------
# 1. Connect
# ----------------------------------------------------------------------
URL = "postgresql+psycopg://diploma:diploma@localhost:5500/vdataset"

print("Started", flush=True)
t0 = time.perf_counter()
try:
    engine = create_engine(
        URL,
        pool_pre_ping=True,
        connect_args={
            "connect_timeout": 5,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 5,
            "keepalives_count": 3,
        },
    )
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))  # quick round‑trip
    print(f"✅  Connected ({time.perf_counter() - t0:.2f}s)", flush=True)
except Exception as exc:
    print(f"\n❌  Connection failed after "
          f"{time.perf_counter() - t0:.2f}s:\n{exc}\n",
          file=sys.stdout, flush=True)
    raise


# ----------------------------------------------------------------------
# 2. ORM mapping & DDL
# ----------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: int | None = Column(BigInteger, primary_key=True)
    source_id: str = Column(Text, unique=True, nullable=False)
    embedding: list[float] = Column(Vector(768), nullable=False)


with engine.begin() as conn:                      # implicit commit/rollback
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    Base.metadata.create_all(conn)
print("✅  Extension ensured + table ready", flush=True)


# ----------------------------------------------------------------------
# 3. Load pickles & insert
# ----------------------------------------------------------------------
def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


paths = ['E:\Diploma\model_training\datasets\processed_chunk_1.pickle',
         'E:\Diploma\model_training\datasets\processed_chunk_2.pickle',
         'E:\Diploma\model_training\datasets\processed_chunk_3.pickle',
         'E:\Diploma\model_training\datasets\processed_chunk_4.pickle',
         'E:\Diploma\model_training\datasets\processed_chunk_5.pickle']
BATCH = 1_000

stmt = insert(Document).on_conflict_do_nothing(index_elements=["source_id"])

with Session(engine) as session:
    for p, i in  enumerate(paths):
        chunk = load_pickle(p)
        stem = Path(p).stem
        rows = [{'source_id': f"{stem}_{i}",  "embedding":item} for item in chunk]      

        for start in tqdm(range(0, len(rows), BATCH), desc=str(p)):
            subset = rows[start:start + BATCH]
            session.execute(stmt, subset)
            print(start, str(p))
                         
        session.commit()
        print("Loaded chunk:", stem)
print("✅  All chunks ingested!")
