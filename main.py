from fastapi import FastAPI
from app.config.db import Base, async_engine
from app.routers import characters, chat, auth
from app.models import user, character, conversation, message
from fastapi.middleware.cors import CORSMiddleware
import asyncio

app = FastAPI(title="AI Fantasy Chat App")

# Create a startup event handler
@app.on_event("startup")
async def startup():
    # Create tables
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(characters.router, prefix="/characters", tags=["characters"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI + SQLite App!"}