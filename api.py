from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from main import RecipeRAGSystem

rag_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_system
    print("🚀 服务启动：初始化 RAG 系统（构建知识库，约 1 分钟）...")
    rag_system = RecipeRAGSystem()
    rag_system.initialize_system()
    rag_system.build_knowledge_base()
    print("✅ 初始化完成，可接受请求")
    yield
    print("🛑 服务关闭")

app = FastAPI(title="尝尝咸淡 RAG 系统", lifespan=lifespan)

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    answer = rag_system.ask_question(req.question)
    return AskResponse(answer=answer)