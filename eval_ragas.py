import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from main import RecipeRAGSystem

rag_system = RecipeRAGSystem()
rag_system.initialize_system()
rag_system.build_knowledge_base()

TEST_QUESTIONS = [
    # ============ 做法查询（8 条，均命中数据中真实菜品） ============
    "白灼虾怎么做？",
    "葱油桂鱼需要哪些食材？",
    "拔丝土豆的制作步骤是什么？",
    "番茄牛肉蛋花汤怎么做？",
    "炒方便面的做法", 
    "茶叶蛋怎么煮？",
    "奥利奥冰淇淋怎么做？",
    "巴基斯坦牛肉咖喱需要什么食材？",

    # ============ 模糊推荐（4 条，考验查询重写） ============
    "推荐一个简单的素菜",
    "有什么饮品推荐吗？",
    "适合新手做的汤",
    "想做个甜品，有什么推荐？",

    # ============ 库外问题（3 条，预期拒答，单独分析） ============
    "重庆大学推免政策是什么？",
    "今天股市怎么样？",
    "帮我写一段 Python 快速排序的代码",
]

def ask_with_contexts(question: str):
    route_type = rag_system.generation_module.query_router(question)
    rewritten = question if route_type == "list" else rag_system.generation_module.query_rewrite(question)

    filters = rag_system._extract_filters_from_query(question)
    if filters:
        chunks = rag_system.retrieval_module.metadata_filtered_search(
            rewritten, filters, top_k=rag_system.config.top_k
        )
    else:
        chunks = rag_system.retrieval_module.hybrid_search(
            rewritten, top_k=rag_system.config.top_k
        )

    if not chunks:
        return [], "抱歉，没有找到相关的食谱信息。"

    docs = rag_system.data_module.get_parent_documents(chunks)
    contexts = [doc.page_content for doc in docs]

    if route_type == "list":
        answer = rag_system.generation_module.generate_list_answer(question, docs)
    elif route_type == "detail":
        answer = rag_system.generation_module.generate_step_by_step_answer(question, docs)
    else:
        answer = rag_system.generation_module.generate_basic_answer(question, docs)

    return contexts, answer

records = []
for q in TEST_QUESTIONS:
    contexts, answer = ask_with_contexts(q)
    records.append({"question": q, "answer": answer, "contexts": contexts})

dataset = Dataset.from_pandas(pd.DataFrame(records))

judge_llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

result = evaluate(
    dataset,
    metrics=[Faithfulness(), AnswerRelevancy()],
    llm=judge_llm,
    embeddings=embeddings
)

print(result)

df = result.to_pandas()
print(df)
df.to_csv("ragas_results.csv", index=False)