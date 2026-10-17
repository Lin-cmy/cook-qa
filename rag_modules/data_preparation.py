import logging
import hashlib
import re
import uuid
from typing import List, Dict
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

logger = logging.getLogger(__name__)

class DataPreparationModule:
    CATEGORY_MAPPING = {
        "aquatic": "水产",
        "breakfast": "早餐",
        "condiment": "调料",
        "dessert": "甜品",
        "drink": "饮品",
        "meat_dish": "荤菜",
        "semi_finished": "半成品",
        "soup": "汤品",
        "staple": "主食",
        "vegetable_dish": "素菜"
    }

    CATEGORY_LABELS = list(set(CATEGORY_MAPPING.values()))
    DIFFICULTY_LABELS = ["非常简单", "简单", "中等", "困难", "非常困难"]

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.documents: List[Document] = []               # 父文档
        self.chunks: List[Document] = []                  # 子文档
        self.parent_child_map: Dict[str, str] = {} # 子块ID -> 父文档ID的映射

    def load_documents(self) -> List[Document]:
        logger.info(f"正在从 {self.data_path} 中加载文档...")

        documents = []
        data_path_obj = Path(self.data_path)

        for md_file in data_path_obj.rglob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                try:
                    data_root = Path(self.data_path).resolve()
                    relative_path = Path(md_file).resolve().relative_to(data_root).as_posix
                except Exception:
                    relative_path = Path(md_file).as_posix

                parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()

                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(md_file),
                        "parent_id": parent_id,
                        "doc_type": "parent"
                    }
                )

                documents.append(doc)
    
            except Exception:
                logger.warning(f"读取文件 {md_file} 失败: {e}")

        for doc in documents:
            self._enhance_metadata(doc)

        self.documents = documents
        logger.info(f"成功加载 {len(documents)} 个文档")
        return documents

    def _enhance_metadata(self, doc: Document):
        file_path = Path(doc.metadata.get("source", ""))
        path_parts = file_path.parts

        doc.metadata["category"] = "其他"
        for key, value in self.CATEGORY_MAPPING.items():
            if key in path_parts:
                doc.metadata["category"] = value
                break

        doc.metadata["dish_name"] = file_path.stem

        content = doc.page_content
        star_match = re.search(r"★+", content)
        if star_match:
            star_count = len(star_match.group())
            difficulty_map = {
                5: "非常困难",
                4: "困难",
                3: "中等",
                2: "简单",
                1: "非常简单"
            }
            doc.metadata["difficulty"] = difficulty_map.get(star_count, "未知")
        else:
            doc.metadata["difficulty"] = "未知"

    @classmethod
    def get_supported_categories(cls) -> List[str]:
        return cls.CATEGORY_LABELS

    @classmethod
    def get_supported_difficulties(cls) -> List[str]:
        return cls.DIFFICULTY_LABELS

    def chunk_documents(self) -> List[Document]:
        logger.info(f"正在进行 Markdown 结构感知分块")

        if not self.documents:
            raise ValueError("请先加载文档")

        chunks = self._markdown_header_split()

        for i, chunk in enumerate(chunks):
            if "chunk_id" not in chunk.metadata:
                chunk.metadata["chunk_id"] = str(uuid.uuid4())
            chunk.metadata["batch_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        self.chunks = chunks
        logger.info(f"Markdown 分块完成，共生成 {len(chunks)} 个 chunk")
        return chunks

    def _markdown_header_split(self) -> List[Document]:
        headers_to_split_on = [
            ("#", "主标题"),        # 菜品名称
            ("##", "二级标题"),     # 必备原料、计算、操作等
            ("###", "三级标题")     # 简易版本、复杂版本等
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )

        all_chunks = []

        for doc in self.documents:
            try:
                md_chunks = markdown_splitter.split_text(doc.page_content)

                logger.debug(f"文档 {doc.metadata.get("dish_name", "未知")} 分割成 {len(md_chunks)} 个 chunk")

                if len(md_chunks) <= 1:
                    logger.warning(f"文档 {doc.metadata.get("dish_name", "未知")} 未能按标题分割，可能缺少标题结构")

                parent_id = doc.metadata["parent_id"]

                for i, chunk in enumerate(md_chunks):
                    child_id = str(uuid.uuid4())

                    chunk.metadata.update(doc.metadata)
                    chunk.metadata.update({
                        "chunk_id": child_id,
                        "parent_id": parent_id,
                        "doc_type": "child",
                        "chunk_index": i
                    })

                    self.parent_child_map[child_id] = parent_id

                all_chunks.extend(md_chunks)

            except Exception as e:
                logger.warning(f"文档 {doc.metadata.get("source", "未知")} Markdown 分割失败: {e}")
                all_chunks.append(doc)

        logger.info(f"Markdown 结构分割完成，生成 {len(all_chunks)} 个结构化块")
        return all_chunks

    def filter_documents_by_category(self, category: str) -> List[Document]:
        """
        按分类过滤文档

        Args:
            category: 菜品分类

        Returns:
            过滤后的文档列表
        """
        return [doc for doc in self.documents if doc.metadata.get('category') == category]
    
    def filter_documents_by_difficulty(self, difficulty: str) -> List[Document]:
        """
        按难度过滤文档
        
        Args:
            difficulty: 难度等级
            
        Returns:
            过滤后的文档列表
        """
        return [doc for doc in self.documents if doc.metadata.get('difficulty') == difficulty]

    def get_statistics(self) -> Dict[str, Any]:
        if not self.documents:
            return {}

        categories = {}
        diffidulties = {}

        for doc in self.documents:
            category = doc.metadata.get("category", "未知")
            categories[category] = categories.get(category, 0) + 1

            difficulty = doc.metadata.get("difficulty", "未知")
            diffidulties[difficulty] = diffidulties.get(difficulty, 0) + 1

        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "categories": categories,
            "difficulties": diffidulties,
            "avg_chunk_size": sum(chunk.metadata.get("chunk_size", 0) for chunk in self.chunks) / len(self.chunks) if self.chunks else 0
        }

    def export_metadata(self, output_path: str):
        import json

        metadata_list = []
        for doc in self.documents:
            metadata_list.append({
                'source': doc.metadata.get('source'),
                'dish_name': doc.metadata.get('dish_name'),
                'category': doc.metadata.get('category'),
                'difficulty': doc.metadata.get('difficulty'),
                'content_length': len(doc.page_content)
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)
        
        logger.info(f"元数据已导出到: {output_path}")

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        parent_relevance = {}
        parent_docs_map = {}

        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1

                if parent_id not in parent_docs_map:
                    for doc in self.documents:
                        if doc.metadata.get("parent_id") == parent_id:
                            parent_docs_map[parent_id] = doc
                            break

        sorted_parent_ids = sorted(parent_relevance.keys(),
                                   key=lambda x: parent_relevance[x],
                                   reverse=True)

        parent_docs = []
        for parent_id in sorted_parent_ids:
            if parent_id in parent_docs_map:
                parent_docs.append(parent_docs_map[parent_id])

        parent_info = []
        for doc in parent_docs:
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            parent_id = doc.metadata.get("parent_id")
            relevance_count = parent_relevance.get(parent_id, 0)
            parent_info.append(f"{dish_name}({relevance_count}块)")

        logger.info(f"从 {len(child_chunks)} 个子块中找到 {len(parent_docs)} 个去重父文档: {", ".join(parent_info)}")
        return parent_docs