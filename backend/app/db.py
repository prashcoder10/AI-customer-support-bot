# # app/db.py
# from sqlalchemy import create_engine, Column, String
# from sqlalchemy.orm import declarative_base, sessionmaker

# Base = declarative_base()
# # sessions.db stored next to this file (ai_support_backend/app)
# engine = create_engine("sqlite:///./sessions.db", connect_args={"check_same_thread": False})
# SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# class SessionModel(Base):
#     __tablename__ = "sessions"
#     user_id = Column(String, primary_key=True, index=True)

# def init_db():
#     Base.metadata.create_all(bind=engine)

# def get_or_create_session(db, user_id: str):
#     # returns SessionModel instance
#     s = db.query(SessionModel).filter_by(user_id=user_id).first()
#     if s:
#         return s
#     s = SessionModel(user_id=user_id)
#     db.add(s)
#     db.commit()
#     return s
from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()
engine = create_engine("sqlite:///./sessions.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class SessionModel(Base):
    __tablename__ = "sessions"
    user_id = Column(String, primary_key=True, index=True)
    conversation = Column(Text, default="")  # store conversation as JSON or plain text

def init_db():
    Base.metadata.create_all(bind=engine)

def get_or_create_session(db, user_id: str):
    s = db.query(SessionModel).filter_by(user_id=user_id).first()
    if s:
        return s
    s = SessionModel(user_id=user_id, conversation="")
    db.add(s)
    db.commit()
    return s
