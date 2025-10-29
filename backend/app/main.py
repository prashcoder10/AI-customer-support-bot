# # from fastapi import FastAPI, HTTPException
# # from fastapi.middleware.cors import CORSMiddleware
# # from pydantic import BaseModel
# # from .db import init_db, get_session, SessionLocal
# # from .llm import generate_response, retrieve_faq_answer
# # import uuid

# # app = FastAPI(title="AI Support Bot Backend")

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # class Message(BaseModel):
# #     text: str

# # class Session(BaseModel):
# #     user_id: str

# # @app.on_event("startup")
# # def startup_event():
# #     init_db()

# # @app.post("/sessions")
# # def create_session(data: Session):
# #     db = SessionLocal()
# #     db.add(get_session(data.user_id))
# #     db.commit()
# #     return {"session_id": data.user_id}

# # @app.post("/sessions/{session_id}/message")
# # def send_message(session_id: str, message: Message):
# #     db = SessionLocal()
# #     faq_answer, score = retrieve_faq_answer(message.text)
# #     if score < 0.5:
# #         reply = generate_response(message.text)
# #     else:
# #         reply = faq_answer
# #     return {"reply": reply, "score": score}

# # @app.get("/sessions/{session_id}")
# # def get_session_data(session_id: str):
# #     db = SessionLocal()
# #     sess = db.query(get_session).filter_by(user_id=session_id).first()
# #     if not sess:
# #         raise HTTPException(404, "Session not found")
# #     return {"user_id": sess.user_id}
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from .llm import generate_response
# from llm import get_bot_response


# app = FastAPI(title="AI Support Bot Backend")

# # Allow frontend connection
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class ChatRequest(BaseModel):
#     message: str

# @app.post("/chat")
# def chat(request: ChatRequest):
#     user_msg = request.message
#     reply = get_bot_response(user_msg)
#     return {"reply": reply}





# app/main.py
# from fastapi import FastAPI, HTTPException, Depends
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from sqlalchemy.orm import Session

# from . import db, llm
# import uuid
# import os

# app = FastAPI(title="AI Support Bot Backend")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # for dev only; lock down in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class SessionCreate(BaseModel):
#     user_id: str = None

# class Message(BaseModel):
#     text: str

# @app.on_event("startup")
# def startup_event():
#     # initialize DB and models
#     db.init_db()
#     # try to set device automatically: GPU if available else CPU
#     device = None
#     # device parameter is handled in llm.init_all (0 => GPU, -1 => CPU)
#     llm.init_all(device=None)
#     print("Startup complete: FAQ count =", len(llm.FAQ_QUESTIONS) if hasattr(llm, "FAQ_QUESTIONS") else 0)

# # Dependency: DB session
# def get_db():
#     db_session = db.SessionLocal()
#     try:
#         yield db_session
#     finally:
#         db_session.close()

# @app.post("/sessions")
# def create_session(data: SessionCreate, db_session: Session = Depends(get_db)):
#     user_id = data.user_id or str(uuid.uuid4())[:8]
#     sess = db.get_or_create_session(db_session, user_id)
#     return {"session_id": sess.user_id}

# @app.get("/sessions/{session_id}")
# def get_session(session_id: str, db_session: Session = Depends(get_db)):
#     s = db_session.query(db.SessionModel).filter_by(user_id=session_id).first()
#     if not s:
#         raise HTTPException(status_code=404, detail="Session not found")
#     return {"session_id": s.user_id}

# @app.post("/sessions/{session_id}/message")
# def session_message(session_id: str, message: Message, db_session: Session = Depends(get_db)):
#     # ensure session exists
#     s = db.get_or_create_session(db_session, session_id)
#     user_text = message.text
#     # consult FAQ first
#     answer, score = llm.retrieve_faq_answer(user_text)
#     if answer and score >= llm.FAQ_SIM_THRESHOLD:
#         reply = answer
#     else:
#         reply = llm.get_bot_response(user_text)
#     # Here you could also persist conversation to DB (optional)
#     return {"reply": reply, "score": score}

# @app.post("/chat")
# def chat(message: Message):
#     reply = llm.get_bot_response(message.text)
#     return {"reply": reply}





# # app/main.py
# from fastapi import FastAPI, HTTPException, Depends
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from sqlalchemy.orm import Session
# from . import db, llm
# import uuid

# app = FastAPI(title="AI Support Bot Backend")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class SessionCreate(BaseModel):
#     user_id: str = None

# class Message(BaseModel):
#     text: str

# @app.on_event("startup")
# def startup_event():
#     db.init_db()
#     llm.init_all()
#     print("Startup complete: FAQ count =", len(llm.FAQ_QUESTIONS))

# # DB session dependency
# def get_db():
#     db_session = db.SessionLocal()
#     try:
#         yield db_session
#     finally:
#         db_session.close()

# @app.post("/sessions")
# def create_session(data: SessionCreate, db_session: Session = Depends(get_db)):
#     user_id = data.user_id or str(uuid.uuid4())[:8]
#     sess = db.get_or_create_session(db_session, user_id)
#     return {"session_id": sess.user_id}

# @app.get("/sessions/{session_id}")
# def get_session(session_id: str, db_session: Session = Depends(get_db)):
#     s = db_session.query(db.SessionModel).filter_by(user_id=session_id).first()
#     if not s:
#         raise HTTPException(status_code=404, detail="Session not found")
#     return {"session_id": s.user_id}

# @app.post("/sessions/{session_id}/message")
# def session_message(session_id: str, message: Message, db_session: Session = Depends(get_db)):
#     s = db.get_or_create_session(db_session, session_id)
#     reply = llm.get_bot_response(message.text, session_id=session_id)
#     return {"reply": reply}

# @app.post("/chat")
# def chat(message: Message):
#     reply = llm.get_bot_response(message.text)
#     return {"reply": reply}

from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
from app import llm, db

app = FastAPI(title="AI Support Bot Backend")


# ✅ Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SessionCreate(BaseModel):
    user_id: Optional[str] = None


class MessageInput(BaseModel):
    message: str


# In-memory sessions
sessions = {}


@app.on_event("startup")
def startup_event():
    db.init_db()
    llm.load_faqs()
    llm.init_embeddings()
    llm.init_generator()
    print(f"🚀 Startup complete — FAQs loaded: {len(llm.FAQS)}")


@app.post("/sessions")
async def start_session(payload: SessionCreate):
    user_id = payload.user_id or "sess_" + str(uuid.uuid4())[:8]
    sessions.setdefault(user_id, [])
    print(f"✅ Session started: {user_id}")
    return {"session_id": user_id}


@app.post("/sessions/{session_id}/message")
async def send_message(session_id: str, payload: MessageInput):
    user_msg = payload.message.strip()
    print(f"🗣️ User({session_id}): {user_msg}")

    prev_msgs = sessions.get(session_id, [])
    context = "\n".join([m["content"] for m in prev_msgs[-5:]])  # last 5 exchanges

    reply = llm.get_bot_response(user_msg, conversation_context=context)

    # Store conversation
    sessions.setdefault(session_id, []).append({"role": "user", "content": user_msg})
    sessions[session_id].append({"role": "bot", "content": reply})

    print(f"🤖 Bot reply → {reply}")
    return {"reply": reply}
