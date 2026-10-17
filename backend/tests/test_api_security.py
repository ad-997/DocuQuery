import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from app.main import app
from app.services.database import metadata, users, documents, sessions
from app.api.auth import hash_password


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.db = create_engine('sqlite://', poolclass=StaticPool, connect_args={'check_same_thread': False})
        self.addCleanup(self.db.dispose)
        metadata.create_all(self.db)
        for module in ('app.api.auth', 'app.api.documents'):
            self.stack.enter_context(patch(module + '.engine', self.db))
        with self.db.begin() as c:
            for name, role, dept in [('admin', 'admin', 'IT'), ('editor', 'editor', 'HR'), ('viewer', 'viewer', 'HR'), ('other', 'viewer', 'IT')]:
                c.execute(users.insert().values(id=name, username=name, role=role, department=dept,
                    password_hash=hash_password('long-password-123')))
        self.client = TestClient(app)
        self.directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.stack.enter_context(patch('app.api.documents.DATA_DIRECTORY', Path(self.directory)))
        self.stack.enter_context(patch('app.api.documents.corpus_service.refresh'))

    def headers(self, name):
        result = self.client.post('/api/auth/login', json={'username': name, 'password': 'long-password-123'})
        self.assertEqual(result.status_code, 200)
        return {'Authorization': 'Bearer ' + result.json()['access_token']}

    def upload(self, headers):
        with patch('app.api.documents.ingest_document', return_value=[{}]):
            return self.client.post('/api/documents', headers=headers,
                files={'file': ('policy.pdf', b'PDF')}, data={'department': 'HR'})

    def test_authentication_and_expiry(self):
        for method, path in [('get', '/api/auth/me'), ('get', '/api/documents'), ('post', '/api/query')]:
            self.assertEqual(getattr(self.client, method)(path).status_code, 401)
        self.assertEqual(self.client.post('/api/auth/login', json={'username': 'missing', 'password': 'bad'}).status_code, 401)
        headers = self.headers('viewer')
        self.assertEqual(self.client.get('/api/auth/me', headers=headers).json()['role'], 'viewer')
        with self.db.begin() as c:
            c.execute(sessions.update().values(expires_at=time.time() - 1))
        self.assertEqual(self.client.get('/api/auth/me', headers=headers).status_code, 401)

    def test_document_lifecycle_and_roles(self):
        editor, viewer, other = self.headers('editor'), self.headers('viewer'), self.headers('other')
        self.assertEqual(self.upload(viewer).status_code, 403)
        uploaded = self.upload(editor)
        self.assertEqual(uploaded.status_code, 201, uploaded.text)
        doc_id = uploaded.json()['id']
        path = '/api/documents/' + doc_id
        self.assertEqual(self.upload(editor).status_code, 409)
        self.assertEqual(self.client.get('/api/documents', headers=viewer).json()['total'], 1)
        self.assertEqual(self.client.get('/api/documents', headers=other).json()['total'], 0)
        detail = self.client.get(path, headers=viewer)
        self.assertEqual(detail.status_code, 200)
        self.assertNotIn('path', detail.json())
        self.assertEqual(self.client.get(path, headers=other).status_code, 404)
        self.assertEqual(self.client.delete(path, headers=viewer).status_code, 403)
        with patch('app.api.documents.delete_document_chunks') as cleanup:
            self.assertEqual(self.client.delete(path, headers=editor).status_code, 204)
            cleanup.assert_called_once_with(doc_id)
        self.assertFalse(list(Path(self.directory).rglob('*.pdf')))
        self.assertEqual(self.client.get(path, headers=editor).status_code, 404)

    def test_user_management_and_query_allowlist(self):
        admin, viewer = self.headers('admin'), self.headers('viewer')
        body = {'username': 'new', 'password': 'another-password', 'role': 'editor', 'department': 'HR'}
        self.assertEqual(self.client.post('/api/auth/users', headers=viewer, json=body).status_code, 403)
        result = self.client.post('/api/auth/users', headers=admin, json=body)
        self.assertEqual(result.status_code, 201)
        self.assertNotIn('password_hash', result.json())
        self.assertEqual(self.client.post('/api/auth/users', headers=admin, json=body).status_code, 409)
        doc_id = self.upload(admin).json()['id']
        with patch('app.api.query.corpus_service') as corpus, patch('app.api.query.run_rag_pipeline', return_value={'answer': 'ok'}) as run:
            self.assertEqual(self.client.post('/api/query', headers=viewer, json={'question': 'leave'}).status_code, 200)
            self.assertEqual(run.call_args.kwargs['allowed_document_ids'], [doc_id])

    def test_failed_ingestion_is_hidden_from_queries_and_can_be_deleted(self):
        editor = self.headers('editor')
        with patch('app.api.documents.ingest_document', side_effect=ValueError('no text')):
            result = self.client.post('/api/documents', headers=editor,
                files={'file': ('bad.pdf', b'invalid')}, data={'department': 'HR'})
        self.assertEqual(result.status_code, 400)
        records = self.client.get('/api/documents', headers=editor).json()['items']
        self.assertEqual(records[0]['status'], 'failed')
        with patch('app.api.query.corpus_service'), patch('app.api.query.run_rag_pipeline', return_value={}) as run:
            self.client.post('/api/query', headers=editor, json={'question': 'policy'})
            self.assertEqual(run.call_args.kwargs['allowed_document_ids'], [])
        with patch('app.api.documents.delete_document_chunks'):
            self.assertEqual(self.client.delete('/api/documents/' + records[0]['id'], headers=editor).status_code, 204)

    def test_failed_deletion_revokes_access_and_is_retryable(self):
        editor = self.headers('editor')
        doc_id = self.upload(editor).json()['id']
        path = '/api/documents/' + doc_id
        with self.assertLogs(level='ERROR'), patch('app.api.documents.delete_document_chunks', side_effect=RuntimeError('offline')):
            self.assertEqual(self.client.delete(path, headers=editor).status_code, 503)
        self.assertEqual(self.client.get(path, headers=editor).json()['status'], 'deleting')
        from app.api.documents import visible_documents
        self.assertEqual(visible_documents({'role': 'admin'}), [])
        with patch('app.api.documents.delete_document_chunks'):
            self.assertEqual(self.client.delete(path, headers=editor).status_code, 204)

    def test_empty_allowlist_skips_embedding_and_generation_context(self):
        from app.rag.pipeline import run_rag_pipeline
        with patch('app.rag.pipeline.embed_query') as embed, patch('app.rag.pipeline.rerank') as rerank:
            result = run_rag_pipeline('policy', None, allowed_document_ids=[])
        self.assertEqual(result['sources'], [])
        embed.assert_not_called()
        rerank.assert_not_called()

    def test_both_retrievers_filter_before_top_k(self):
        from app.retrieval.sparse import BM25Retriever
        from app.vectorstore import qdrant_store as store
        from qdrant_client import QdrantClient
        chunks = [dict(document_id=name, filename=name+'.pdf', department='HR', file_type='.pdf', page=1, text='policy ' + name) for name in ('secret', 'allowed')]
        sparse = BM25Retriever(chunks)
        self.assertEqual(sparse.search('policy', top_k=1, allowed_document_ids=['allowed'])[0]['chunk']['document_id'], 'allowed')
        client = QdrantClient(':memory:')
        self.addCleanup(client.close)
        vector = np.zeros(384); vector[0] = 1
        with patch.object(store, 'client', client):
            store.upsert_chunks(chunks, [vector, vector])
            self.assertEqual(store.search_qdrant(vector, top_k=1, allowed_document_ids=['allowed'])[0].payload['document_id'], 'allowed')
            self.assertEqual(store.search_qdrant(vector, allowed_document_ids=[]), [])
            store.delete_document_chunks('allowed')
            self.assertEqual(store.search_qdrant(vector, allowed_document_ids=['allowed']), [])
