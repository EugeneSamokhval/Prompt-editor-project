from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DISTANCE_OPS: dict[str, str] = {
    "cosine": "<=>",  # cosine distance (no need for pre-normalised vectors)
    "l2": "<->",       # Euclidean / L2 distance
    "inner": "<#>",   # negative inner-product distance
}


# ---------------------------------------------------------------------------
# Main helper class
# ---------------------------------------------------------------------------
class PromptVectorSearch:
    """Semantic search + prompt-improvement utility on top of pgvector.

    The class assumes that the embeddings already stored in the database were
    produced by **sentence-transformers/all-MiniLM-L6-v2** *without* explicit
    L2 normalisation – exactly the configuration shown in the original loading
    script. At query time we therefore **do not normalise** either, ensuring a
    perfect match between query and corpus spaces.
    """

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def __init__(
        self,
        db_url: str,
        *,
        table: str = "documents",
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embed_fn: Callable[[str], Sequence[float]] | None = None,
        embed_dim: int = 384,
        metric: str = "cosine",
        device: str | None = None,
        normalize: bool = False,
    ) -> None:
        """Create a new *PromptVectorSearch* instance.

        Parameters
        ----------
        db_url : str
            SQLAlchemy URL, e.g. ``postgresql+psycopg://user:pass@host:port/db``.
        table : str, optional
            Name of the table containing at least the columns ``source_id`` and
            ``embedding`` (Vector). Optionally a ``text`` column may hold the
            original chunk text. Default ``documents``.
        embed_model_name : str, optional
            Model to load via *sentence-transformers*. Default is exactly the
            model used during ingestion (all-MiniLM-L6-v2).
        embed_fn : Callable[[str], Sequence[float]] | None, optional
            Custom embedding callable. If supplied, *embed_model_name* is
            ignored and *device/normalize* settings are irrelevant.
        embed_dim : int, optional
            Dimensionality of embeddings stored in the DB. Default 384.
        metric : {"cosine","l2","inner"}, optional
            Distance metric/operator to use. Default "cosine" (<=>).
        device : {"cuda","cpu",None}, optional
            Device string passed to *SentenceTransformer*. ``None`` lets the
            library decide. Using ``"cuda"`` will move the model to GPU if
            available.
        normalize : bool, optional
            Whether to L2-normalise embeddings aterce time. **Must match the
            setting used at index-time** – here *False*.
        """
        self.engine = create_engine(db_url, pool_pre_ping=True, future=True)
        self.table = table
        self.embed_dim = embed_dim
        self.normalize = normalize

        if metric not in DISTANCE_OPS:
            raise ValueError(f"Unsupported metric {metric!r}; choose one of {list(DISTANCE_OPS)}")
        self._distance_op = DISTANCE_OPS[metric]

        if embed_fn is not None:
            self.embed = embed_fn  # type: ignore[assignment]
        else:
            # Always prefer Torch device selection – falls back gracefully.
            self._model = SentenceTransformer(embed_model_name, device=device)
            # Force FP32 on CPU for consistency; Torch handles casting on GPU.
            if device is None and not torch.cuda.is_available():
                torch.set_default_dtype(torch.float32)

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def embed(self, text_: str) -> List[float]:  # noqa: D401
        """Convert a single *text_* string to a float32 embedding."""
        vec: np.ndarray = self._model.encode(
            text_,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        if vec.shape[0] != self.embed_dim:  # pragma: no cover
            raise ValueError(f"Expected dim={self.embed_dim}, got {vec.shape[0]}")
        return vec.astype(np.float32).tolist()
    
    def index_vectors(self):
        connection =  self.engine.connect()
        connection.execute(
        text(f"""
            CREATE INDEX IF NOT EXISTS
                {LEVEL_TABLE_NAME_PREFIX}{level}_embedding_hnsw_index 
                ON {SCHEMA_NAME}.{LEVEL_TABLE_NAME_PREFIX}{level} USING hnsw (embedding vector_cosine_ops);
        """)
    )
    # ------------------------------------------------------------------
    # Core vector search
    # ------------------------------------------------------------------
    def search(
        self,
        prompt: str,
        *,
        k: int = 10,
        return_text: bool = True,
    ) -> List[Dict]:
        """Retrieve *k* nearest neighbours of *prompt* from the database."""
        query_vec = self.embed(prompt)

        cols = f"source_id, embedding {self._distance_op} ARRAY[:vec] AS distance"
        if return_text:
            cols += ", text"  # may silently omit if column absent

        # Convert query_vec to a string representation for pgvector
        vec_str = ','.join(map(str, query_vec))

        sql = text(
            f"""
            SELECT {cols}
              FROM {self.table}
          ORDER BY embedding {self._distance_op} ARRAY[:vec]
             LIMIT :k
            """
        )
        with self.engine.connect() as conn:
            rows = conn.execute(sql, {"vec": vec_str, "k": k}).mappings().all()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Prompt augmentation
    # ------------------------------------------------------------------
    def improve_prompt(
        self,
        prompt: str,
        *,
        k: int = 5,
        separator: str = "\n\n---\n\n",
        max_chars: int | None = 6_000,
    ) -> str:
        """Augment *prompt* with the *k* most relevant chunks via vector search."""
        hits = self.search(prompt, k=k, return_text=True)

        if any(hit.get("text") for hit in hits):
            context = separator.join(
                f"[{hit['source_id']}]\n{hit.get('text', '')}".strip() for hit in hits
            )
        else:
            context = separator.join(hit["source_id"] for hit in hits)

        improved = f"{prompt.strip()}{separator}{context.strip()}"
        if max_chars and len(improved) > max_chars:
            improved = improved[: max_chars]
        return improved

if __name__ == "__main__":
    import os
    searcher = PromptVectorSearch(
        db_url=os.getenv('VECTOR_DB_URL'),
        device='cuda',
        normalize=False,
    )
    print(searcher.improve_prompt('Anime aesthetic, view from the foot of volcano, gigantic volcano, a town around the volcano, asian architecture, bright day, fantasy settings, perfect, atmospheric perspective, perspective, tall shot, UHD, masterpiece, accurate, super detail, high details, high quality, award winning, 16k'))