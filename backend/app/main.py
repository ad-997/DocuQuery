from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.services.database import initialize
from app.api.auth import router as auth_router
from app.api.query import router as query_router
from app.api.documents import router as documents_router


@asynccontextmanager
async def lifespan(app):
    initialize()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="DocuQuery API",
    version="1.0.0"
)

app.include_router(auth_router)
app.include_router(query_router)
app.include_router(documents_router)


@app.get("/")
def home():
    return {
        "message": "DocuQuery backend is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }
