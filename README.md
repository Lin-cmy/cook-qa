# cook-qa

基于 HowToCook 开源菜谱数据（323 篇 Markdown、10 个分类）的检索增强问答系统（RAG）。
链路覆盖数据准备 → 索引构建 → 检索优化 → 生成集成 → 服务化，并配有一套可复跑的评测。

## 链路

```
问题
  │
  ├─ 前置路由（1 次 LLM 调用）
  │     ├─ out_of_domain → 直接返回拒答话术（不进检索与生成）
  │     ├─ list          → 检索后直接拼装结果（不调生成）
  │     └─ detail/general→ 查询重写 → 检索 → 生成
  │
  ├─ 检索：BM25（稀疏）+ FAISS（稠密）双路召回，RRF 融合，父子文档回取整篇菜谱
  │
  └─ 生成：deepseek-chat，严格约束只使用检索到的上下文作答
```

关键设计：

| 设计 | 解决的问题 |
|---|---|
| 父子文档映射 | 小块检索提升命中精度，命中后回取整篇菜谱，避免"检索命中但上下文残缺" |
| BM25 + 向量双路 RRF 融合 | 单一检索在关键词匹配与语义匹配上各有盲区 |
| 前置领域路由 | 库外问题在路由阶段拦截，单次问答 LLM 调用从 3 次降到 1 次 |
| 流式输出 + TTFT 埋点 | 流式场景下用户感知的是首字延迟，需与端到端耗时分开测量 |

## 快速开始

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY=sk-...        # 或写入 .env

python main.py                        # CLI 交互式问答
python api.py                         # FastAPI 服务，默认 :8000，/docs 查看接口
```

容器化：

```bash
docker build -t cook-qa .
docker run -d -p 8000:8000 -e DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY cook-qa
curl http://127.0.0.1:8000/health
```

首次启动会构建知识库（323 篇加载 + 分块 + 1764 条向量化，约 8.5s）；
索引落盘到 `vector_index/` 后，二次启动直接加载（约 36ms）。

## 评测

```bash
python eval_routing.py            # 路由：拦截率 / 误拦率，约 1 分钟，无需构建知识库
python eval_ragas.py --repeat 3   # 生成质量：RAGAs 判分 + 分组延迟，约 3 分钟
```

自建 22 条评测集（12 条库内 + 10 条库外，库外含 3 条提示注入）。
"回答质量"与"该不该回答"是两类问题，因此拆成两套指标：

- **库内**：RAGAs 判生成质量（faithfulness / answer_relevancy）
- **库外**：规则断言判拦截（正确答案是唯一的拒答话术，属于二分类，不需要 LLM 裁判）

| 指标 | 结果 |
|---|---|
| faithfulness（12 条库内） | 加入严格上下文约束 + 前置路由后 **0.85 → 0.96**（95%CI ±0.030） |
| answer_relevancy | 0.70 → 0.53（结构化长回答会被该指标系统性惩罚，属于已知取舍） |
| 库外拦截 | **10/10**，库内误拦 **0/12** |
| detail 路由延迟 | 首字 P50 **1.75s**、端到端 P50 **4.26s**（3 次串行 LLM 调用） |
| list 路由延迟 | **0.81s**（省去重写与生成两次调用） |

样本为 12 条 × 3 轮 = 36 次采样；36 个样本不足以报 P95，故只报 P50 与最大值。

## 已知局限

1. **近域问题可能编造**：卡路里、糖尿病人能否食用等走 `general` 路由进入检索+生成，
   而知识库不含营养数据，存在编造风险。需要补"数据不足即声明缺失"的规则或 `context_recall` 门槛。
2. **只评了生成层**：没有 `context_precision` / `context_recall`，无法把"检索错"与"生成编"分开归因。
3. **judge 与被测同为 deepseek-chat**：存在同源偏好，换用其他模型交叉验证更可信。
4. **样本量小**：覆盖度不足以谈分组显著性，`general` 路由当前 0 条样本。
5. **延迟只测热态**：不含冷启动（索引加载 + 首次连接）。

## 目录

```
config.py                 配置（模型、top_k、路径）
main.py                   RAG 主链路，run_query() 返回结构化 RAGResult
api.py                    FastAPI 服务
rag_modules/
  data_preparation.py     数据加载与父子分块
  index_construction.py   向量索引构建与落盘
  retrieval_optimizatin.py BM25 + 向量双路召回、RRF 融合、查询重写
  generation_integration.py 路由、流式生成、上下文约束
eval_datasets.py          评测集（唯一真相源）
eval_routing.py           路由评测
eval_ragas.py             生成质量评测 + 延迟分组统计
EVAL.md                   评测设计与结果的完整说明
```

## 数据来源

菜谱数据来自开源项目 [HowToCook](https://github.com/Anduin2017/HowToCook)，仅用于学习用途。
