"""库内生成质量评测：只对"应该被回答"的问题跑 RAGAs，并按 route_type 分组统计延迟。

与旧版的关键差别：
1. 数据集只剩库内 12 条 —— 拒答样本不再混进 faithfulness/answer_relevancy 的均值
   （旧版把 3 条库外拒答算进去，它们的 faithfulness 0.4/0.5/0.71、relevancy 0 直接把总分拉低）；
2. 走 `RecipeRAGSystem.run_query()` 而不是在评测脚本里重写一遍链路 —— 旧版自己重新调
   router/rewrite/retrieve/generate，绕过了 out_of_domain 的提前 return，于是"库外拦截"
   这条路径从来没被评到过（口径漂移的根源）；
3. 用 **流式模式**跑，一次请求同时得到 TTFT 与端到端耗时 —— 非流式只能测到"全部生成完"，
   测不出用户真正感知的首字延迟；
4. 延迟按 route_type 分组统计 —— detail（3 次串行 LLM 调用）和 list（纯字符串拼装）
   是两种完全不同的请求，混在一起算的均值既不代表 detail 用户，也不代表 list 用户；
5. 样本量不足时拒绝报 P95 —— 12 个样本算第 95 百分位，结果基本等于最大值，是假精度。

拒答能力请跑 eval_routing.py —— 那是二分类问题，不该用在这里。

运行：
    python eval_ragas.py                 # 质量 + 1 轮延迟采样
    python eval_ragas.py --repeat 5      # 质量只评第 1 轮，延迟汇总 5 轮（12*5=60 次采样）
输出：
    ragas_results_in_domain.csv
"""
import argparse
import os
import statistics
import sys
from collections import defaultdict
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

# P95 的最低样本量。低于这个数就退化为报 max —— 12 个样本的第 95 百分位
# 落在最大的那一两个值上，看起来精确，其实只是最大值换了个名字。
P95_MIN_N = 30


def run_query_streaming(rag_system: RecipeRAGSystem, question: str):
    """以流式模式跑一次链路，返回 (RAGResult, 完整回答文本)。

    为什么用流式跑评测：用户感知的是首字延迟，非流式只能测到"全部生成完"。
    流式一次调用同时给出 TTFT 与总时长，两个数来自同一次请求，
    不会因为两次请求之间的缓存/负载差异而不可比。
    """
    result = rag_system.run_query(question, stream=True, verbose=False)

    # list 路由是纯字符串拼装、拦截分支是固定话术，都不返回生成器。
    # 此时 ttft_ms 与 latency_ms 在 run_query 内部已写好。
    if isinstance(result.answer, str):
        return result, result.answer

    chunks = []
    for chunk in result.answer:      # 消费生成器：包装器在此写入 ttft_ms / latency_ms
        chunks.append(chunk)
    return result, "".join(chunks)


def collect(rag_system: RecipeRAGSystem, rounds: int = 1):
    """跑 rounds 轮。

    第 1 轮的答案进 RAGAs（质量只需要评一次），
    所有轮的延迟进统计（延迟是随机变量，重复采样才有分布）。
    """
    records = []
    leaked = []
    latency_samples = defaultdict(list)   # route_type -> [(ttft_ms, total_ms, calls)]

    for r in range(rounds):
        print(f"\n--- 第 {r + 1}/{rounds} 轮 ---")
        for case in IN_DOMAIN_CASES:
            result, text = run_query_streaming(rag_system, case["q"])

            # 库内问题被误拦 = 路由器的假阳性。这类样本不能进 RAGAs（没有上下文可评），
            # 单列出来计数，最后由 eval_routing.py 统一汇报误拦率。
            if result.intercepted or text == OUT_OF_DOMAIN_REPLY:
                if r == 0:
                    leaked.append(case["q"])
                    print(f"⚠️ 误拦（库内问题被判为库外）: {case['q']}")
                continue

            if not result.contexts:
                if r == 0:
                    print(f"⚠️ 未检索到上下文，跳过: {case['q']}")
                continue

            latency_samples[result.route_type].append(
                (result.ttft_ms, result.latency_ms, result.llm_calls)
            )

            if r == 0:
                records.append(
                    {
                        "user_input": case["q"],
                        "response": text,
                        "retrieved_contexts": result.contexts,
                        "route_type": result.route_type,
                        "llm_calls": result.llm_calls,
                        "ttft_ms": result.ttft_ms,
                        "latency_ms": result.latency_ms,
                    }
                )
                print(
                    f"✅ [{result.route_type:<7}] TTFT {result.ttft_ms:>5}ms  "
                    f"总耗时 {result.latency_ms:>5}ms  calls={result.llm_calls}  {case['q']}"
                )

    return records, leaked, latency_samples


def pctl(values, q: float):
    """线性插值分位数；样本不足时返回 None（表示"这个数不该报"）。"""
    if not values:
        return None
    if q > 0.5 and len(values) < P95_MIN_N:
        return None
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def ms(v):
    """延迟格式化：None -> 不报（样本不足），避免把缺失值打成 0 误导人看。"""
    return "  n/a" if v is None else f"{v:.0f}ms"


def print_grouped(latency_samples, scores: dict):
    """按 route_type 输出延迟与质量；最后单独给一行混算值作为对照。"""
    print("\n===== 按 route_type 分组（延迟必须分组看，混算的数字不代表任何真实用户）=====")
    header = (
        f"{'路由':<8}{'n':>4}{'faith':>8}{'relev':>8}"
        f"{'TTFT P50':>11}{'TTFT max':>11}{'耗时 P50':>11}{'耗时 max':>10}{'耗时 P95':>10}{'调用/次':>9}"
    )
    print(header)
    print("-" * 88)

    all_ttft, all_total, all_calls = [], [], []
    for route in sorted(latency_samples):
        samples = latency_samples[route]
        ttft = [s[0] for s in samples]
        total = [s[1] for s in samples]
        calls = [s[2] for s in samples]
        all_ttft += ttft
        all_total += total
        all_calls += calls

        sc = scores.get(route, {})
        f = sc.get("faithfulness")
        a = sc.get("answer_relevancy")
        print(
            f"{route:<8}{len(samples):>4}"
            f"{(f'{f:.3f}' if f is not None else '-'):>8}"
            f"{(f'{a:.3f}' if a is not None else '-'):>8}"
            f"{ms(pctl(ttft, 0.5)):>11}{ms(max(ttft) if ttft else None):>11}"
            f"{ms(pctl(total, 0.5)):>11}{ms(max(total) if total else None):>10}"
            f"{ms(pctl(total, 0.95)):>10}"
            f"{statistics.mean(calls):>9.2f}"
        )

    # 混算行：只为说明"为什么不能混算"，不作为结论使用
    if len(latency_samples) > 1:
        print("-" * 88)
        print(
            f"{'混算(勿用)':<8}{len(all_total):>4}{'-':>8}{'-':>8}"
            f"{ms(pctl(all_ttft, 0.5)):>11}{ms(max(all_ttft)):>11}"
            f"{ms(pctl(all_total, 0.5)):>11}{ms(max(all_total)):>10}"
            f"{ms(pctl(all_total, 0.95)):>10}{statistics.mean(all_calls):>9.2f}"
        )
        print(
            "  混算行的均值会被快请求（list 纯拼接）带低，"
            "真实 detail 用户的体验比这一行差，面试只报分组值。"
        )


def speech_line(latency_samples, scores: dict):
    """生成一句能直接口述的结论：分组 + 归因 + 样本量诚实说明。"""
    parts = []
    for route in sorted(latency_samples):
        samples = latency_samples[route]
        total = [s[1] for s in samples]
        ttft = [s[0] for s in samples]
        parts.append(
            f"{route} 路由 n={len(samples)}，TTFT P50 {pctl(ttft, 0.5) / 1000:.1f}s、"
            f"端到端 P50 {pctl(total, 0.5) / 1000:.1f}s、最大 {max(total) / 1000:.1f}s"
        )
    n_total = sum(len(v) for v in latency_samples.values())
    p95_missing = n_total < P95_MIN_N
    tail = (
        f"；{n_total} 次采样不足 {P95_MIN_N} 条，不报 P95，只报 P50 与最大值"
        if p95_missing
        else f"；{n_total} 次采样，端到端 P95 {pctl([s[1] for v in latency_samples.values() for s in v], 0.95) / 1000:.1f}s"
    )
    return "；".join(parts) + tail


def main():
    parser = argparse.ArgumentParser(description="库内生成质量 + 分组延迟评测")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="延迟采样重复轮数（质量只评第 1 轮）。12 条问题 x 3 轮 = 36 次采样才够报 P95",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat 必须 >= 1")

    rag_system = RecipeRAGSystem()
    rag_system.initialize_system()
    rag_system.build_knowledge_base()

    records, leaked, latency_samples = collect(rag_system, rounds=args.repeat)
    if not records:
        raise SystemExit("没有任何可用样本，请检查知识库与路由配置")

    print(
        f"\n有效样本 {len(records)} 条"
        + (f"，误拦 {len(leaked)} 条（已剔除，请跑 eval_routing.py 复核）" if leaked else "，无误拦")
        + f"；延迟采样 {sum(len(v) for v in latency_samples.values())} 次"
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

    # 把链路指标（路由/延迟/调用次数）并回结果表，方便一张表看完"质量 + 成本"
    for col in ("route_type", "llm_calls", "ttft_ms", "latency_ms"):
        df[col] = [r[col] for r in records]

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(df[["user_input", "route_type", "faithfulness", "answer_relevancy", "ttft_ms", "latency_ms", "llm_calls"]])

    # 质量指标也按路由分组：detail 与 list 的生成方式完全不同，
    # 把两者的分数平均到一起，等于报一个没有任何对应实体的数字。
    scores = {}
    for route in sorted({r["route_type"] for r in records}):
        sub = df[df["route_type"] == route]
        scores[route] = {
            "faithfulness": sub["faithfulness"].mean(),
            "answer_relevancy": sub["answer_relevancy"].mean(),
        }

    print_grouped(latency_samples, scores)

    print("\n===== 质量汇总（仅库内样本）=====")
    print(f"样本数               : {len(df)}")
    print(f"faithfulness 均值    : {df['faithfulness'].mean():.3f}")
    print(f"answer_relevancy 均值: {df['answer_relevancy'].mean():.3f}")
    sd = df["faithfulness"].std()
    if len(df) > 1:
        # 只有 12 条样本时，均值的小数点第二位是不可信的 —— 给出置信区间，
        # 面试被问"你这个 0.96 准不准"时能直接答"95%CI ±0.04"。
        print(
            f"faithfulness 95%CI   : ±{1.96 * sd / (len(df) ** 0.5):.3f}"
            f"（sd={sd:.3f}, n={len(df)}；样本越少区间越宽，别把小数点后两位当结论）"
        )
    print(f"\n明细已写入 {OUT_CSV}")

    print("\n----- 可直接写进简历 / 面试口述 -----")
    print(
        f"自建 {len(df)} 条库内评测集，faithfulness {df['faithfulness'].mean():.2f}"
        f"（answer_relevancy {df['answer_relevancy'].mean():.2f} 作为参考指标）；"
        + speech_line(latency_samples, scores)
        + "。"
    )


if __name__ == "__main__":
    main()
