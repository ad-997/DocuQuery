"""Cache the lexical corpus; publish a complete index on refresh."""
from pathlib import Path
from threading import Lock

from app.ingestion.pipeline import load_corpus_chunks
from app.retrieval.sparse import BM25Retriever

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"


class CorpusService:
    def __init__(self):
        self._retriever = None
        self._lock = Lock()

    @property
    def retriever(self):
        with self._lock:
            if self._retriever is None:
                self._refresh()
            return self._retriever

    @property
    def chunks(self):
        return self.retriever.chunks

    def _refresh(self):
        retriever = BM25Retriever(load_corpus_chunks(DATA_DIRECTORY))
        self._retriever = retriever

    def refresh(self):
        with self._lock:
            self._refresh()


corpus_service = CorpusService()
