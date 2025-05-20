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
EMBED_DIM  = 384
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

with engine.begin() as conn:
    conn.execute(text("DROP TABLE IF EXISTS documents CASCADE;"))
    Base.metadata.create_all(conn) 
    
class Document(Base):
    __tablename__ = "documents"

    id: int | None = Column(BigInteger, primary_key=True)
    source_id: str = Column(Text, unique=True, nullable=False)
    embedding: list[float] = Column(Vector(EMBED_DIM), nullable=False)


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
    
def to_vector(x: np.ndarray | list[float], *, method="mean") -> list[float]:
    arr = np.asarray(x, dtype=float)

    if arr.ndim == 1:            # already 1-D
        if arr.size != EMBED_DIM:
            raise ValueError(f"unexpected length {arr.size}, expected EMBED_DIM")
        return arr.tolist()

    if arr.ndim == 2 and arr.shape[1] == EMBED_DIM:
        arr = arr.mean(axis=0)
        return arr.tolist()

    raise ValueError(f"unsupported shape {arr.shape}; expected (n,EMBED_DIM) or (EMBED_DIM,)")

paths = ['E:\Diploma\model_training\datasets\processed_chunk_1.pickle',
         'E:\Diploma\model_training\datasets\processed_chunk_2.pickle',
         'E:\Diploma\model_training\datasets\processed_chunk_3.pickle',
         'E:\Diploma\model_training\datasets\processed_chunk_4.pickle',
         'E:\Diploma\model_training\datasets\processed_chunk_5.pickle']
BATCH = 1_000

stmt = insert(Document).on_conflict_do_nothing(index_elements=["source_id"])

row_counts = {p: len(load_pickle(p)) for p in paths}
total_rows = sum(row_counts.values())

with Session(engine) as session, \
     tqdm(total=total_rows, desc="TOTAL rows",  unit="row",  position=0) as total_bar, \
     tqdm(paths,        desc="FILES",       unit="file", position=1, leave=False) as file_bar:

    for path in file_bar:
        stem  = Path(path).stem
        chunk = load_pickle(path)

        # show a dedicated bar for the current file
        with tqdm(total=len(chunk),
                  desc=f"{stem}",   
                  unit="row",
                  position=2,
                  leave=False) as file_rows_bar:

            # build the INSERT rows once per file
            rows = [{"source_id": f"{stem}_{i}", "embedding": to_vector(emb)}
                    for i, emb in enumerate(chunk)]

            for start in range(0, len(rows), BATCH):
                subset = rows[start:start + BATCH]
                session.execute(stmt, subset)
                session.commit()

                # advance both bars
                batch_len = len(subset)
                file_rows_bar.update(batch_len)
                total_bar.update(batch_len)

print("✅  All chunks ingested!")