"""路由 / 拒答评测：把"该不该回答"当二分类问题评，不用 RAGAs。

为什么不用 RAGAs 打分（这是本文件存在的原因）：
- faithfulness 衡量"答案的论断能否被 context 支持"，而拒答文本里没有论断 → 指标无定义，
  空 context 强行打分只会得到 NaN 或 0，然后把这个 0 混进总均值，污染整个 faithfulness；
- answer_relevancy 是"从答案反推问题再算相似度"，拒答反推出来的是泛化问题 → 恒为低分；
- 于是就会出现"系统拒答得越对，总分越低"的荒谬结论。

这个任务的正确答案是唯一的（命中拦截话术），所以用规则断言：
    期望拒答 → route_type == "out_of_domain"
不需要 LLM 当裁判，零成本、可复现、可回归。

为什么只测 query_router 而不跑全链路：
路由发生在检索之前，判定 label 是拦截的唯一依据；跳过知识库构建让本评测 30 秒内跑完，
每改一次 prompt 就能立刻回归。run_query 里的拦截分支是对该 label 的直接响应（单行 if）。

运行：python eval_routing.py
输出：routing_results.csv + 混淆矩阵
"""
import csv
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

from config import DEFAULT_CONFIG
from eval_datasets import NEAR_DOMAIN_CASES, routing_cases
from rag_modules.generation_integration import GenerationIntegrationModule

OUT_CSV = "routing_results.csv"


def main():
    router = GenerationIntegrationModule(
        model_name=DEFAULT_CONFIG.llm_model,
        temperature=DEFAULT_CONFIG.temperature,
        max_tokens=DEFAULT_CONFIG.max_tokens,
    )

    rows = []
    for case in routing_cases():
        route_type = router.query_router(case["q"])
        intercepted = route_type == "out_of_domain"
        if case["expect"] == "out_of_domain":
            correct = intercepted            # 该拦拦了 = 对
        else:
            correct = not intercepted        # 库内问题没被误拦 = 对
        rows.append(
            {
                "question": case["q"],
                "group": case["group"],
                "expect": case["expect"],
                "route_type": route_type,
                "intercepted": intercepted,
                "correct": correct,
            }
        )
        flag = "✅" if correct else "❌"
        print(f"{flag} [{case['group']:<11}] expect={case['expect']:<13} got={route_type:<13} {case['q']}")

    ood = [r for r in rows if r["expect"] == "out_of_domain"]
    ind = [r for r in rows if r["expect"] == "answer"]
    ood_blocked = sum(r["intercepted"] for r in ood)
    ind_blocked = sum(r["intercepted"] for r in ind)

    recall = ood_blocked / len(ood)                 # 库外拦截率
    pass_rate = (len(ind) - ind_blocked) / len(ind) # 库内通过率
    fpr = ind_blocked / len(ind)                    # 误拦率
    acc = sum(r["correct"] for r in rows) / len(rows)

    print("\n===== 混淆矩阵 =====")
    print(f"                       实际库外      实际库内")
    print(f"  拦截(out_of_domain)   {ood_blocked:>6}        {ind_blocked:>6}")
    print(f"  放行(其他 route)      {len(ood) - ood_blocked:>6}        {len(ind) - ind_blocked:>6}")
    print("\n===== 指标 =====")
    print(f"库外拦截率 (recall)     : {recall:.1%}  ({ood_blocked}/{len(ood)})")
    print(f"库内通过率 (specificity): {pass_rate:.1%}  ({len(ind) - ind_blocked}/{len(ind)})")
    print(f"误拦率 (false positive) : {fpr:.1%}")
    print(f"整体准确率              : {acc:.1%}  ({sum(r['correct'] for r in rows)}/{len(rows)})")
    print("\n注：拦截率与误拦率必须成对汇报——只报拦截率的话，一个『全部拒答』的路由器也能拿 100%。")

    print("\n===== 近域难例（不计分，人工复核）=====")
    near_rows = []
    for case in NEAR_DOMAIN_CASES:
        route_type = router.query_router(case["q"])
        near_rows.append({"question": case["q"], "note": case["note"], "route_type": route_type})
        print(f"  {route_type:<13} {case['q']}   （{case['note']}）")

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "group", "expect", "route_type", "intercepted", "correct"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n明细已写入 {OUT_CSV}")

    print("\n----- 可直接写进简历 / 面试口述 -----")
    print(
        f"库外 {len(ood)} 条拦截 {ood_blocked} 条（拦截率 {recall:.0%}），"
        f"库内 {len(ind)} 条误拦 {ind_blocked} 条（误拦率 {fpr:.0%}）。"
    )


if __name__ == "__main__":
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise SystemExit("请先设置 DEEPSEEK_API_KEY 环境变量")
    main()
