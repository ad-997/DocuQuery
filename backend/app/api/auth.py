"""Password authentication with expiring, hashed opaque bearer tokens."""
import hashlib
import hmac
import secrets
import time
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy import select
from app.services.database import engine, users, sessions

router = APIRouter(prefix='/api/auth', tags=['Authentication'])
bearer = HTTPBearer(auto_error=False)

def hash_password(password):
    salt = secrets.token_hex(16)
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()
    return f'{salt}:{digest}'

def verify_password(password, stored):
    salt, digest = stored.split(':')
    candidate = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()
    return hmac.compare_digest(candidate, digest)

def public_user(row):
    return {key: row[key] for key in ('id', 'username', 'role', 'department')}

def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
    error = HTTPException(401, 'Invalid or expired credentials.', headers={'WWW-Authenticate': 'Bearer'})
    if credentials is None:
        raise error
    token_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()
    with engine.connect() as connection:
        row = connection.execute(select(users).join(sessions, users.c.id == sessions.c.user_id).where(
            sessions.c.token_hash == token_hash, sessions.c.expires_at > time.time())).mappings().first()
    if row is None:
        raise error
    return public_user(row)

def administrator(user=Depends(current_user)):
    if user['role'] != 'admin':
        raise HTTPException(403, 'Administrator role required.')
    return user

class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=1024)

@router.post('/login')
def login(body: LoginRequest):
    with engine.begin() as connection:
        row = connection.execute(select(users).where(users.c.username == body.username)).mappings().first()
        # Perform the same expensive verification for unknown users.
        stored = row['password_hash'] if row else '00' * 16 + ':' + '00' * 64
        if not verify_password(body.password, stored) or row is None:
            raise HTTPException(401, 'Invalid username or password.', headers={'WWW-Authenticate': 'Bearer'})
        token = secrets.token_urlsafe(32)
        connection.execute(sessions.insert().values(token_hash=hashlib.sha256(token.encode()).hexdigest(),
            user_id=row['id'], expires_at=time.time() + 3600))
    return {'access_token': token, 'token_type': 'bearer', 'expires_in': 3600}

@router.get('/me')
def me(user=Depends(current_user)):
    return user

class UserRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100, pattern=r'^\S+$')
    password: str = Field(min_length=12, max_length=1024)
    role: Literal['admin', 'editor', 'viewer'] = 'viewer'
    department: str

@router.post('/users', status_code=201)
def create_user(body: UserRequest, admin=Depends(administrator)):
    from uuid import uuid4
    from sqlalchemy.exc import IntegrityError
    from app.api.documents import DEPARTMENTS
    department = DEPARTMENTS.get(body.department.strip().lower())
    if department is None:
        raise HTTPException(400, 'Invalid department.')
    values = dict(id=str(uuid4()), username=body.username, password_hash=hash_password(body.password),
                  role=body.role, department=department)
    try:
        with engine.begin() as connection:
            connection.execute(users.insert().values(**values))
    except IntegrityError:
        raise HTTPException(409, 'Username already exists.')
    return public_user(values)
