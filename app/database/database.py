"""Подключение к базе данных"""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.database.models import Base


# Создание движка базы данных
engine = create_engine(
    settings.database_url,
    poolclass=StaticPool,
    connect_args={
        "check_same_thread": False,  # Для SQLite
    },
    echo=settings.debug  # Логирование SQL запросов в debug режиме
)

# Создание сессии
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    """Создать все таблицы в базе данных"""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Получить сессию базы данных
    
    Yields:
        Session: Сессия SQLAlchemy
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> Generator[Session, None, None]:
    """
    Асинхронная версия получения сессии БД
    
    Yields:
        Session: Сессия SQLAlchemy
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class DatabaseManager:
    """Менеджер базы данных"""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
    
    def create_all_tables(self):
        """Создать все таблицы"""
        Base.metadata.create_all(bind=self.engine)
    
    def drop_all_tables(self):
        """Удалить все таблицы (осторожно!)"""
        Base.metadata.drop_all(bind=self.engine)
    
    def get_session(self) -> Session:
        """Получить новую сессию"""
        return self.SessionLocal()
    
    def close_session(self, session: Session):
        """Закрыть сессию"""
        session.close()


# Глобальный экземпляр менеджера БД
db_manager = DatabaseManager() 