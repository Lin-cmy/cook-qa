import logging
import hashlib
import re
from typing import List, Dict
from pathlib import Path

from langchain_core.documents import Document

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