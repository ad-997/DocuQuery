import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from app.main import app
from app.api.auth import current_user
from app.ingestion.chunker import chunk_pages
from app.ingestion.pdf_loader import load_pdf
from app.retrieval.sparse import BM25Retriever, search_bm25
from app.reranking.reranker import rerank
from app.generation.generator import generate_answer


class PipelineRegressionTests(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides[current_user] = lambda: {'id': 'test', 'role': 'admin', 'department': 'HR'}
        self.addCleanup(app.dependency_overrides.clear)
        self.client = TestClient(app)

    def test_health_and_blank_question_without_models(self):
        self.assertEqual(self.client.get('/health').json(), {'status': 'healthy'})
        self.assertEqual(self.client.post('/api/query', json={'question': '  '}).status_code, 400)

    def test_query_passes_cached_retriever(self):
        with patch('app.api.query.corpus_service') as corpus, patch('app.api.query.visible_documents', return_value=[]), patch('app.api.query.run_rag_pipeline', return_value={'answer': 'ok'}) as run:
            response = self.client.post('/api/query', json={'question': ' question '})
            self.assertEqual(response.status_code, 200)
            run.assert_called_once_with(question='question', bm25_retriever=corpus.retriever, allowed_document_ids=[])

    def test_query_error_does_not_expose_exception(self):
        with self.assertLogs(level='ERROR'), patch('app.api.query.corpus_service'), patch('app.api.query.run_rag_pipeline', side_effect=RuntimeError('secret')):
            response = self.client.post('/api/query', json={'question': 'question'})
        self.assertEqual(response.status_code, 500)
        self.assertNotIn('secret', response.text)

    def test_bm25_empty_and_legacy_wrapper(self):
        self.assertEqual(BM25Retriever([]).search('leave'), [])
        self.assertEqual(BM25Retriever([{'text': '!!!'}]).search('leave'), [])
        chunks = [{'text': 'parental leave'}, {'text': 'security password'}, {'text': 'expense policy'}]
        self.assertEqual(search_bm25('parental', chunks)[0]['chunk'], chunks[0])
        self.assertEqual(BM25Retriever(chunks).search(''), [])

    def test_chunk_parameters_and_whitespace(self):
        for size, overlap in [(0, 0), (5, 5), (5, -1)]:
            with self.assertRaises(ValueError):
                chunk_pages([], {}, size, overlap)
        metadata = dict(document_id='a', filename='a.pdf', department='HR', file_type='.pdf')
        self.assertEqual(chunk_pages([{'page': 1, 'text': '   '}], metadata), [])
        chunks = chunk_pages([{'page': 2, 'text': 'abcdefgh'}], metadata, 5, 2)
        self.assertEqual(chunks[0]['text'], 'abcde')
        self.assertEqual(chunks[1]['text'], 'defgh')
        self.assertEqual(chunks[1]['page'], 2)

    def test_missing_pdf_text(self):
        with patch('app.ingestion.pdf_loader.PdfReader') as reader:
            from types import SimpleNamespace
            reader.return_value.pages = [SimpleNamespace(extract_text=lambda: None)]
            self.assertEqual(load_pdf('unused'), [{'page': 1, 'text': ''}])

    def test_empty_evidence_skips_models_and_generation(self):
        with patch('app.reranking.reranker.get_model') as model, patch('app.generation.generator.Groq') as groq:
            self.assertEqual(rerank('question', []), [])
            self.assertIn('not find enough information', generate_answer('question', ''))
            model.assert_not_called()
            groq.assert_not_called()

    def test_upload_validation_and_cleanup(self):
        from sqlalchemy import create_engine
        from sqlalchemy.pool import StaticPool
        from app.services.database import metadata
        database = create_engine('sqlite://', poolclass=StaticPool, connect_args={'check_same_thread': False})
        metadata.create_all(database)
        self.addCleanup(database.dispose)
        with tempfile.TemporaryDirectory() as directory, patch('app.api.documents.engine', database), patch('app.api.documents.DATA_DIRECTORY', Path(directory)):
            self.assertEqual(self.client.post('/api/documents', files={'file': ('a.txt', b'hello')}, data={'department': 'HR'}).status_code, 400)
            with patch('app.api.documents.ingest_document', side_effect=ValueError('invalid')):
                response = self.client.post('/api/documents', files={'file': ('invalid.pdf', b'invalid')}, data={'department': 'HR'})
            self.assertEqual(response.status_code, 400)
            self.assertFalse((Path(directory) / 'HR/a.pdf').exists())
            with patch('app.api.documents.ingest_document', return_value=[{}]), patch('app.api.documents.corpus_service.refresh'):
                response = self.client.post('/api/documents', files={'file': ('a.pdf', b'pdf')}, data={'department': 'HR'})
                self.assertEqual(response.status_code, 201)
                response = self.client.post('/api/documents', files={'file': ('a.pdf', b'other')}, data={'department': 'HR'})
                self.assertEqual(response.status_code, 409)
                self.assertEqual(next(Path(directory).rglob('a.pdf')).read_bytes(), b'pdf')


class VectorStoreTests(unittest.TestCase):
    def test_upsert_and_search_in_isolated_local_database(self):
        import numpy as np
        from qdrant_client import QdrantClient
        from app.vectorstore import qdrant_store as store
        client = QdrantClient(":memory:")
        chunk = dict(document_id="HR-TEST", filename="test.pdf", department="HR",
                     file_type=".pdf", page=1, text="parental leave")
        vector = np.zeros(384)
        vector[0] = 1
        try:
            with patch.object(store, 'client', client):
                self.assertEqual(store.search_qdrant(vector), [])
                store.upsert_chunks([chunk], [vector])
                store.upsert_chunks([chunk], [vector])
                self.assertEqual(client.count(store.COLLECTION_NAME).count, 1)
                results = store.search_qdrant(vector)
                self.assertEqual(results[0].payload, chunk)
                self.assertAlmostEqual(results[0].score, 1)
        finally:
            client.close()
