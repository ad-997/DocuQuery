"""Relational metadata storage. Set DATABASE_URL for PostgreSQL."""
import os
from pathlib import Path
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Float, UniqueConstraint
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parents[3]
engine = create_engine(os.getenv('DATABASE_URL', f'sqlite:///{ROOT / "data/docuquery.db"}'), pool_pre_ping=True)
metadata = MetaData()
users = Table('users', metadata,
    Column('id', String, primary_key=True), Column('username', String, unique=True, nullable=False),
    Column('password_hash', String, nullable=False), Column('role', String, nullable=False),
    Column('department', String, nullable=False))
sessions = Table('sessions', metadata,
    Column('token_hash', String, primary_key=True), Column('user_id', String, nullable=False),
    Column('expires_at', Float, nullable=False))
documents = Table('documents', metadata,
    Column('id', String, primary_key=True), Column('filename', String, nullable=False),
    Column('department', String, nullable=False), Column('path', String, unique=True, nullable=False),
    Column('owner_id', String, nullable=False), Column('chunks_created', Integer, nullable=False),
    Column('created_at', Float, nullable=False), Column('status', String, nullable=False),
    UniqueConstraint('department', 'filename', name='uq_document_department_filename'))

def initialize():
    metadata.create_all(engine)
