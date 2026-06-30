from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from narcoticsense.database.models import Base


def create_session_factory(database_url: str = "sqlite:///narcoticsense.db"):
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
