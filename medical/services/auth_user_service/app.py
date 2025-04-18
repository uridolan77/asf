# FastAPI entrypoint for Auth/User Service
from fastapi import FastAPI
from .routes import router

app = FastAPI(title="Auth/User Service")
app.include_router(router, prefix="/api/auth")
