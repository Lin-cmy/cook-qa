"""库内生成质量评测：只对"应该被回答"的问题跑 RAGAs。

与旧版的关键差别：
1. 数据集只剩库内 12 条 —— 拒答样本不再混进 faithfulness/answer_relevancy 的均值
   （旧版把 3 条库外拒答算进去，它们的 faithfulness 0.4/0.5/0.71、relevancy 0 直接把总分拉低）；
2. 走 `RecipeRAGSystem.run_query()` 而不是在评测脚本里重写一遍链路 —— 旧版自己重新调
   router/rewrite/retrieve/generate，绕过了 out_of_domain 的提前 return，于是"库外拦截"
   这条路径从来没被评到过（口径漂移的根源）；
3. 顺带记录 route_type / latency_ms / llm_calls，一份脚本同时产出质量、延迟、调用次数三类数字。

拒答能力请跑 eval_routing.py —— 那是二分类问题，不该用在这里。

运行：python eval_ragas.py
输出：ragas_results_in_domain.csv
"""
import os
import statistics
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import AnswerRelevancy, Faithfulness
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from eval_datasets import IN_DOMAIN_CASES
from main import OUT_OF_DOMAIN_REPLY, RecipeRAGSystem

OUT_CSV = "ragas_results_in_domain.csv"


def collect(rag_system: RecipeRAGSystem):
    """逐条跑真实链路，收集 (问题, 回答, 上下文, 延迟, 调用次数)。"""
    records = []
    leaked = []
    for case in IN_DOMAIN_CASES:
        result = rag_system.run_query(case["q"], verbose=False)

        # 库内问题被误拦 = 路由器的假阳性。这类样本不能进 RAGAs（没有上下文可评），
        # 单列出来计数，最后由 eval_routing.py 统一汇报误拦率。
        if result.intercepted or result.answer == OUT_OF_DOMAIN_REPLY:
            leaked.append(case["q"])
            print(f"⚠️ 误拦（库内问题被判为库外）: {case['q']}")
            continue

        if not result.contexts:
            print(f"⚠️ 未检索到上下文，跳过: {case['q']}")
            continue

        records.append(
            {
                "user_input": case["q"],
                "response": result.answer,
                "retrieved_contexts": result.contexts,
                "route_type": result.route_type,
                "llm_calls": result.llm_calls,
                "latency_ms": result.latency_ms,
            }
        )
        print(f"✅ [{result.route_type:<7}] {result.latency_ms:>5}ms  calls={result.llm_calls}  {case['q']}")

    return records, leaked


def main():
    rag_system = RecipeRAGSystem()
    rag_system.initialize_system()
    rag_system.build_knowledge_base()

    records, leaked = collect(rag_system)
    if not records:
        raise SystemExit("没有任何可用样本，请检查知识库与路由配置")

    print(
        f"\n有效样本 {len(records)} 条"
        + (f"，误拦 {len(leaked)} 条（已剔除，请跑 eval_routing.py 复核）" if leaked else "，无误拦")
    )

    dataset = Dataset.from_pandas(
        pd.DataFrame(
            [
                {
                    "user_input": r["user_input"],
                    "response": r["response"],
                    "retrieved_contexts": r["retrieved_contexts"],
                }
                for r in records
            ]
        )
    )

    judge_llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0,
        timeout=120,
        max_retries=3,
    )
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

    result = evaluate(
        dataset,
        metrics=[Faithfulness(), AnswerRelevancy()],
        llm=judge_llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )

    df = result.to_pandas()

    # 把链路指标（延迟/调用次数）并回结果表，方便一张表看完"质量 + 成本"
    for col in ("route_type", "llm_calls", "latency_ms"):
        df[col] = [r[col] for r in records]

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(df)

    lat = sorted(r["latency_ms"] for r in records)
    p50 = statistics.median(lat)
    p95 = lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))]
    avg_calls = statistics.mean(r["llm_calls"] for r in records)

    print("\n===== 汇总（仅库内样本）=====")
    print(f"样本数               : {len(df)}")
    print(f"faithfulness 均值    : {df['faithfulness'].mean():.3f}")
    print(f"answer_relevancy 均值: {df['answer_relevancy'].mean():.3f}")
    print(f"延迟 P50 / P95       : {p50:.0f}ms / {p95:.0f}ms")
    print(f"平均 LLM 调用次数    : {avg_calls:.2f} 次/问答")
    print(f"\n明细已写入 {OUT_CSV}")

    print("\n----- 可直接写进简历 / 面试口述 -----")
    print(
        f"自建 {len(df)} 条库内评测集，faithfulness {df['faithfulness'].mean():.2f}"
        f"（answer_relevancy {df['answer_relevancy'].mean():.2f} 作为参考指标），"
        f"端到端 P95 {p95 / 1000:.1f}s，单次问答平均 {avg_calls:.1f} 次 LLM 调用。"
    )


if __name__ == "__main__":
    main()
