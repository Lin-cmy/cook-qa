import os
import logging
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    def __init__(self, model_name: str = "deepseek-chat", temperature: float = 0.1, max_tokens: int = 2048):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.setup_llm()

    def setup_llm(self):
        logger.info(f"正在初始化大模型 {self.model_name}")

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")

        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

        logger.info("LLM初始化完成")

    def generate_basic_answer(self, query: str, context_docs: List[Document])-> str:
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

请提供详细、实用的回答。如果信息不足，请诚实说明。
回答必须严格基于提供的食谱信息，不得补充食谱中未出现的内容；无法确定的细节宁可省略，也不要猜测或编造。"""),
            ("human", """"用户问题: {question}

相关食谱信息:
{context}""")
        ])

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response

    def generate_step_by_step_answer(self, query: str, context_docs: List[Document]) -> str:
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位专业的烹饪导师。请根据食谱信息，为用户提供详细的分步骤指导。

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 🥘 菜品介绍
[简要介绍菜品特点和难度]

## 🛒 所需食材
[列出主要食材和用量]

## 👨‍🍳 制作步骤
[详细的分步骤说明，每步包含具体操作和大概所需时间]

## 💡 制作技巧
[仅在有实用技巧时包含。优先使用原文中的实用技巧，如果原文的"附加内容"与烹饪无关或为空，可以基于制作步骤总结关键要点，或者完全省略此部分]

注意：
- 根据实际内容灵活调整结构
- 不要强行填充无关内容或重复制作步骤中的信息
- 重点突出实用性和可操作性
- 如果没有额外的技巧要分享，可以省略制作技巧部分
- 回答必须严格基于提供的食谱信息，不得补充食谱中未出现的内容；无法确定的细节宁可省略，也不要猜测或编造"""),
            ("human", """用户问题: {question}

相关食谱信息:
{context}""")
        ])

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response

    def query_rewrite(self, query: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个智能查询分析助手。请分析用户的查询，判断是否需要重写以提高食谱搜索效果。

分析规则：
1. **具体明确的查询**（直接返回原查询）：
   - 包含具体菜品名称：如"宫保鸡丁怎么做"、"红烧肉的制作方法"
   - 明确的制作询问：如"蛋炒饭需要什么食材"、"糖醋排骨的步骤"
   - 具体的烹饪技巧：如"如何炒菜不粘锅"、"怎样调制糖醋汁"

2. **模糊不清的查询**（需要重写）：
   - 过于宽泛：如"做菜"、"有什么好吃的"、"推荐个菜"
   - 缺乏具体信息：如"川菜"、"素菜"、"简单的"
   - 口语化表达：如"想吃点什么"、"有饮品推荐吗"

重写原则：
- 保持原意不变
- 增加相关烹饪术语
- 优先推荐简单易做的
- 保持简洁性

示例：
- "做菜" → "简单易做的家常菜谱"
- "有饮品推荐吗" → "简单饮品制作方法"
- "推荐个菜" → "简单家常菜推荐"
- "川菜" → "经典川菜菜谱"
- "宫保鸡丁怎么做" → "宫保鸡丁怎么做"（保持原查询）
- "红烧肉需要什么食材" → "红烧肉需要什么食材"（保持原查询）

请输出最终查询（如果不需要重写就返回原查询）"""),
            ("human", """原始查询: {query}""")
        ])

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query).strip()

        if response != query:
            logger.info(f"查询已重写: '{query}' -> '{response}'")
        else:
            logger.info(f"查询无需重写: '{query}'")

        return response

    def query_router(self, query: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """根据用户的问题，将其分类为以下四种类型之一：

1. 'out_of_domain' - 与烹饪、菜谱、食材完全无关的问题
   例如：帮我写一段代码、今天股市怎么样、重庆大学的推免政策

2. 'list' - 用户想要获取菜品列表或推荐，只需要菜名
   例如：推荐几个素菜、有什么川菜、给我3个简单的菜

3. 'detail' - 用户想要具体的制作方法或详细信息
   例如：宫保鸡丁怎么做、制作步骤、需要什么食材

4. 'general' - 其他一般性问题
   例如：什么是川菜、制作技巧、营养价值

请只返回分类结果：list、detail、general 或 out_of_domain"""),
            ("human", """用户问题: {query}""")
        ])

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        result = chain.invoke(query).strip().lower()

        if result in ["list", "detail", "general", "out_of_domain"]:
            return result
        else:
            return "general"

    def generate_list_answer(self, query: str, context_docs: List[Document]) -> str:
        if not context_docs:
            return "抱歉，没有找到相关的菜品信息"

        dish_names = []
        for doc in context_docs:
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            if dish_name not in dish_names:
                dish_names.append(dish_name)

        if len(dish_names) == 1:
            return f"为您推荐: {dish_names[0]}"
        elif len(dish_names) <= 3:
            return f"为您推荐以下菜品：\n" + "\n".join([f"{i+1}. {name}" for i, name in enumerate(dish_names)])
        else:
            return f"为您推荐以下菜品：\n" + "\n".join([f"{i+1}. {name}" for i, name in enumerate(dish_names[:3])]) + f"\n\n还有其他 {len(dish_names)-3} 道菜品可供选择。"

    def generate_basic_answer_stream(self, query: str, context_docs: List[Document]):
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

请提供详细、实用的回答。如果信息不足，请诚实说明。
回答必须严格基于提供的食谱信息，不得补充食谱中未出现的内容；无法确定的细节宁可省略，也不要猜测或编造。"""),
            ("human", """用户问题: {question}

相关食谱信息:
{context}""")
        ])

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def generate_step_by_step_answer_stream(self, query: str, context_docs: List[Document]):
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位专业的烹饪导师。请根据食谱信息，为用户提供详细的分步骤指导。

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 🥘 菜品介绍
[简要介绍菜品特点和难度]

## 🛒 所需食材
[列出主要食材和用量]

## 👨‍🍳 制作步骤
[详细的分步骤说明，每步包含具体操作和大概所需时间]

## 💡 制作技巧
[仅在有实用技巧时包含。优先使用原文中的实用技巧，如果原文的"附加内容"与烹饪无关或为空，可以基于制作步骤总结关键要点，或者完全省略此部分]

注意：
- 根据实际内容灵活调整结构
- 不要强行填充无关内容或重复制作步骤中的信息
- 重点突出实用性和可操作性
- 如果没有额外的技巧要分享，可以省略制作技巧部分
- 回答必须严格基于提供的食谱信息，不得补充食谱中未出现的内容；无法确定的细节宁可省略，也不要猜测或编造"""),
            ("human", """用户问题: {question}

相关食谱信息:
{context}""")
        ])

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def _build_context(self, docs: List[Document], max_length: int = 2000) -> str:
        if not docs:
            return "暂无相关食谱信息"

        context_parts = []
        current_length = 0

        for i, doc in enumerate(docs, 1):
            metadata_info = f"【食谱 {i}】"
            if "dish_name" in doc.metadata:
                metadata_info += f" {doc.metadata["dish_name"]}"
            if "category" in doc.metadata:
                metadata_info += f" | 分类：{doc.metadata["category"]}"
            if "difficulty" in doc.metadata:
                metadata_info += f" | 难度：{doc.metadata["difficulty"]}"

            doc_text = f"{metadata_info}\n{doc.page_content}\n"

            if current_length + len(doc_text) > max_length:
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        divider = "\n" + "=" * 50 + "\n"
        return divider + divider.join(context_parts)