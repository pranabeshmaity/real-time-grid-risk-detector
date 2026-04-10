from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from app.core.config import settings

engine = None
async_session_maker = None
Base = declarative_base()

async def init_db():
    global engine, async_session_maker
    engine = None
    async_session_maker = None

async def close_db():
    if engine:
        await engine.dispose()

async def get_session() -> AsyncSession:
    return None
