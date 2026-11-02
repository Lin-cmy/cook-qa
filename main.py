"""
RAG系统主程序
"""

import os
import sys
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule
)

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 库外问题统一拒答话术。抽成常量是为了让评测能断言"是否命中拦截"，
# 而不是靠 LLM 判分去猜这句回答"算不算拒答"。
OUT_OF_DOMAIN_REPLY = (
    "抱歉，我是菜谱助手，只能回答与烹饪、菜品、食材相关的问题。"
    "您可以问我菜品做法、食材搭配或食谱推荐。"
)


@dataclass
class RAGResult:
    """一次问答的完整结构化结果。

    为什么要有这个类：评测脚本此前把 ask_question 的链路复制了一遍，
    于是"服务实际走的路径"和"评测走的路径"会各自漂移（库外拦截就是这样没被评到的）。
    把链路收敛到 run_query 一处，评测/API/埋点都消费这个对象，就不会再漂。
    """

    question: str
    route_type: str = "general"          # out_of_domain / list / detail / general
    rewritten_query: str = ""
    filters: Dict[str, Any] = field(default_factory=dict)
    child_chunks: List[Any] = field(default_factory=list)   # 命中的子块
    parent_docs: List[Any] = field(default_factory=list)    # 回取的父文档（生成上下文）
    contexts: List[str] = field(default_factory=list)       # 父文档正文，直接喂给 RAGAs
    answer: Any = ""                                        # stream=True 时是生成器
    intercepted: bool = False            # 是否被前置拦截（未进检索与生成）
    llm_calls: int = 0                   # 本次消耗的 LLM 调用次数（成本代理指标）
    latency_ms: int = 0                  # 端到端耗时；stream=True 时为到首个 chunk 的时间


class RecipeRAGSystem:
    """食谱RAG系统主类"""

    def __init__(self, config: RAGConfig = None):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        # 检查数据路径
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        # 检查API密钥
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")
    
    def initialize_system(self):
        """初始化所有模块"""
        print("🚀 正在初始化RAG系统...")

        # 1. 初始化数据准备模块
        print("初始化数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_path)

        # 2. 初始化索引构建模块
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )

        # 3. 初始化生成集成模块
        print("🤖 初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        print("✅ 系统初始化完成！")
    
    def build_knowledge_base(self):
        """构建知识库"""
        print("\n正在构建知识库...")

        # 1. 尝试加载已保存的索引
        vectorstore = self.index_module.load_index()

        if vectorstore is not None:
            print("✅ 成功加载已保存的向量索引！")
            # 仍需要加载文档和分块用于检索模块
            print("加载食谱文档...")
            self.data_module.load_documents()
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()
        else:
            print("未找到已保存的索引，开始构建新索引...")

            # 2. 加载文档
            print("加载食谱文档...")
            self.data_module.load_documents()

            # 3. 文本分块
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()

            # 4. 构建向量索引
            print("构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)

            # 5. 保存索引
            print("保存向量索引...")
            self.index_module.save_index()

        # 6. 初始化检索优化模块
        print("初始化检索优化...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        # 7. 显示统计信息
        stats = self.data_module.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   菜品分类: {list(stats['categories'].keys())}")
        print(f"   难度分布: {stats['difficulties']}")

        print("✅ 知识库构建完成！")
    
    def run_query(self, question: str, stream: bool = False, verbose: bool = True) -> RAGResult:
        """走完一次问答的完整链路，返回结构化结果。

        这是链路的唯一真相来源：ask_question()、评测脚本、API 都调它。
        评测要 contexts 就取 result.contexts，要延迟就取 result.latency_ms，
        不必再自己重写一遍"路由→重写→检索→生成"（重写就是评测口径漂移的根源）。

        Args:
            question: 用户问题
            stream: 是否流式生成（此时 answer 是生成器，latency_ms 记首 chunk 时间）
            verbose: 是否打印过程日志（评测时关掉）

        Returns:
            RAGResult
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        t0 = time.perf_counter()
        result = RAGResult(question=question)

        if verbose:
            print(f"\n❓ 用户问题: {question}")

        # 1. 路由（1 次 LLM 调用）
        result.route_type = self.generation_module.query_router(question)
        result.llm_calls = 1
        if verbose:
            print(f"🎯 查询类型: {result.route_type}")

        # 1.5 领域拦截：非烹饪问题直接拒答，不进检索与生成（省 2 次 LLM 调用）
        if result.route_type == "out_of_domain":
            result.intercepted = True
            result.answer = OUT_OF_DOMAIN_REPLY
            result.latency_ms = int((time.perf_counter() - t0) * 1000)
            return result

        # 2. 查询重写（列表查询保持原样，省 1 次调用）
        if result.route_type == "list":
            result.rewritten_query = question
            if verbose:
                print(f"📝 列表查询保持原样: {question}")
        else:
            if verbose:
                print("🤖 智能分析查询...")
            result.rewritten_query = self.generation_module.query_rewrite(question)
            result.llm_calls += 1

        # 3. 检索（有元数据约束走过滤检索，否则混合召回）
        result.filters = self._extract_filters_from_query(question)
        if result.filters:
            if verbose:
                print(f"应用过滤条件: {result.filters}")
            result.child_chunks = self.retrieval_module.metadata_filtered_search(
                result.rewritten_query, result.filters, top_k=self.config.top_k
            )
        else:
            result.child_chunks = self.retrieval_module.hybrid_search(
                result.rewritten_query, top_k=self.config.top_k
            )

        # 4. 检索为空：生成层兜底
        if not result.child_chunks:
            result.answer = "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"
            result.latency_ms = int((time.perf_counter() - t0) * 1000)
            return result

        # 5. 子块 -> 父文档（小粒度检索保精度，大粒度上下文保完整）
        result.parent_docs = self.data_module.get_parent_documents(result.child_chunks)
        result.contexts = [doc.page_content for doc in result.parent_docs]
        if verbose:
            names = [doc.metadata.get("dish_name", "未知菜品") for doc in result.parent_docs]
            print(f"找到 {len(result.child_chunks)} 个子块，回取 {len(result.parent_docs)} 篇菜谱: {', '.join(names)}")

        # 6. 生成（list 分支是纯字符串拼装，不消耗 LLM 调用）
        if result.route_type == "list":
            result.answer = self.generation_module.generate_list_answer(question, result.parent_docs)
        elif stream:
            generator = (
                self.generation_module.generate_step_by_step_answer_stream(question, result.parent_docs)
                if result.route_type == "detail"
                else self.generation_module.generate_basic_answer_stream(question, result.parent_docs)
            )
            result.answer = generator
            result.latency_ms = int((time.perf_counter() - t0) * 1000)
            return result
        else:
            result.answer = (
                self.generation_module.generate_step_by_step_answer(question, result.parent_docs)
                if result.route_type == "detail"
                else self.generation_module.generate_basic_answer(question, result.parent_docs)
            )
        result.llm_calls += 1
        result.latency_ms = int((time.perf_counter() - t0) * 1000)
        return result

    def ask_question(self, question: str, stream: bool = False):
        """兼容旧接口：只返回答案字符串。

        需要检索上下文 / 路由结果 / 延迟时请用 run_query()。
        注意 stream=True 时返回值是生成器，此时 RAGResult.answer 也是生成器。
        """
        result = self.run_query(question, stream=stream)
        return result.answer

    def _extract_filters_from_query(self, query: str) -> dict:
        """
        从用户问题中提取元数据过滤条件
        """
        filters = {}
        # 分类关键词
        category_keywords = DataPreparationModule.get_supported_categories()
        for cat in category_keywords:
            if cat in query:
                filters['category'] = cat
                break

        # 难度关键词
        difficulty_keywords = DataPreparationModule.get_supported_difficulties()
        for diff in sorted(difficulty_keywords, key=len, reverse=True):
            if diff in query:
                filters['difficulty'] = diff
                break

        return filters
    
    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("🍽️  尝尝咸淡RAG系统 - 交互式问答  🍽️")
        print("=" * 60)
        print("💡 解决您的选择困难症，告别'今天吃什么'的世纪难题！")
        
        # 初始化系统
        self.initialize_system()
        
        # 构建知识库
        self.build_knowledge_base()
        
        print("\n交互式问答 (输入'退出'结束):")
        
        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break
                
                # 询问是否使用流式输出
                stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                use_stream = stream_choice != 'n'

                print("\n回答:")
                if use_stream:
                    # 流式输出
                    for chunk in self.ask_question(user_input, stream=True):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    # 普通输出
                    answer = self.ask_question(user_input, stream=False)
                    print(f"{answer}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")
        
        print("\n感谢使用尝尝咸淡RAG系统！")


def main():
    """主函数"""
    try:
        # 创建RAG系统
        rag_system = RecipeRAGSystem()
        
        # 运行交互式问答
        rag_system.run_interactive()
        
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")

if __name__ == "__main__":
    main()
