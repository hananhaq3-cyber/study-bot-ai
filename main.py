from fastapi import FastAPI
from pydantic import BaseModel
import os
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()

app = FastAPI()

# ---------- MongoDB Connection ----------
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["study_bot_ai"]
collection = db["chats"]

# ---------- Request Schema ----------
class ChatRequest(BaseModel):
    user_id: str
    message: str

# ---------- LLM ----------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# ---------- Routes ----------
@app.get("/")
def home():
    return {"message": "Study Bot AI with memory is running 🚀"}

@app.post("/chat")
def chat(request: ChatRequest):
    user_id = request.user_id
    user_message = request.message

    # ---------- System Prompt (Study Bot Logic) ----------
    messages = [
        SystemMessage(content="""
You are StudyBot, an AI assistant designed to help students learn.

Your responsibilities:
- Help with academic subjects like mathematics, science, history, programming, and general education.
- Provide accurate and clear explanations.
- Explain concepts step-by-step when needed.
- Use simple language that students can understand.
- Encourage learning and curiosity.

If a user asks something unrelated to studying, politely guide them back to educational topics.
""")
    ]

    # ---------- Get Previous Messages ----------
    history = collection.find(
        {"user_id": user_id}
    ).sort("timestamp", -1).limit(10)

    for chat in reversed(list(history)):
        messages.append(HumanMessage(content=chat["user_message"]))
        messages.append(AIMessage(content=chat["bot_reply"]))

    # ---------- Add New User Message ----------
    messages.append(HumanMessage(content=user_message))

    # ---------- Ask LLM ----------
    response = llm.invoke(messages)

    # ---------- Save Conversation ----------
    collection.insert_one({
        "user_id": user_id,
        "user_message": user_message,
        "bot_reply": response.content,
        "timestamp": datetime.utcnow()
    })

    return {
        "user_id": user_id,
        "user_message": user_message,
        "bot_reply": response.content
    }