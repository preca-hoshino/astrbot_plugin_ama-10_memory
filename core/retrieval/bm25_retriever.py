"""
BM25检索器 - 基于SQLite FTS5 或 PostgreSQL tsvector 的稀疏检索
实现简洁的BM25检索功能,用于MemoryEngine的混合检索
"""

import json
from dataclasses import dataclass
from typing import Any

from astrbot.api import logger

from ..processors.text_processor import TextProcessor
from ...storage.pg_connection import get_pool


@dataclass
class BM25Result:
    """BM25检索结果"""

    doc_id: int
    score: float
    content: str
    metadata: dict[str, Any]


class BM25Retriever:
    """
    文档路 BM25 关键词检索器

    使用SQLite FTS5或PostgreSQL tsvector实现BM25算法的全文检索。
    主要特性:
    1. 使用TextProcessor进行中文分词和停用词过滤
    2. 支持通过metadata过滤session_id和persona_id
    3. BM25分数自动归一化到[0,1]区间
    """

    def __init__(
        self,
        db_path: str,
        text_processor: TextProcessor,
        config: dict[str, Any] | None = None,
    ):
        """
        初始化BM25检索器

        Args:
            db_path: SQLite数据库路径
            text_processor: 文本处理器实例
            config: 配置字典(可选)
        """
        self.db_path = db_path
        self.text_processor = text_processor
        self.config = config or {}
        self.doc_table = "documents"

    async def initialize(self):
        """
        初始化
        PostgreSQL: tsv 列由触发器自动更新，无需手动创建 FTS 表
        """
        pass

    async def add_document(
        self, doc_id: int, content: str, metadata: dict[str, Any] | None = None
    ):
        """添加文档到BM25索引 (PG: tsv 列由触发器自动更新)"""
        pass

    async def search(
        self,
        query: str,
        limit: int = 50,
        session_id: str | None = None,
        persona_id: str | None = None,
    ) -> list[BM25Result]:
        """
        执行BM25搜索

        Args:
            query: 查询字符串
            limit: 返回结果数量
            session_id: 会话ID过滤(可选)
            persona_id: 人格ID过滤(可选)

        Returns:
            BM25Result列表,按归一化分数降序排列
        """
        if not query or not query.strip():
            return []

        # 预处理查询（异步卸载 jieba 分词到线程池）
        tokens = await self.text_processor.tokenize_async(query, remove_stopwords=True)
        if not tokens:
            return []

        return await self._search_pg(tokens, limit, session_id, persona_id)

    async def _search_pg(
        self,
        tokens: list[str],
        limit: int,
        session_id: str | None = None,
        persona_id: str | None = None,
    ) -> list[BM25Result]:
        """PG tsvector BM25 搜索"""
        pool = get_pool()
        ts_query = " | ".join(tokens)

        filters: list[str] = []
        params: list[Any] = [ts_query]
        idx = 2
        if session_id is not None:
            filters.append(f"metadata->>'session_id' = ${idx}")
            params.append(session_id)
            idx += 1
        if persona_id is not None:
            filters.append(f"metadata->>'persona_id' = ${idx}")
            params.append(persona_id)
            idx += 1

        where_extra = f"AND {' AND '.join(filters)}" if filters else ""

        sql = f"""
            SELECT id AS doc_id, text, metadata,
                   ts_rank(tsv, to_tsquery('simple', $1)) AS score
            FROM documents
            WHERE tsv @@ to_tsquery('simple', $1) {where_extra}
            ORDER BY score DESC
            LIMIT ${idx}
        """
        params.append(limit)

        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        if not rows:
            return []

        scores = [float(row["score"]) for row in rows]
        max_score = max(scores) if scores else 1.0

        results: list[BM25Result] = []
        for row in rows:
            normalized = float(row["score"]) / max_score if max_score > 0 else 0.0
            metadata = row["metadata"]
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
            results.append(
                BM25Result(
                    doc_id=int(row["doc_id"]),
                    score=normalized,
                    content=row["text"],
                    metadata=metadata or {},
                )
            )
        return results

    async def delete_document(self, doc_id: int) -> bool:
        """从BM25索引删除文档 (PG: tsv 列随文档删除自动处理)"""
        return True

    async def update_document(
        self, doc_id: int, content: str, metadata: dict[str, Any] | None = None
    ) -> bool:
        """更新BM25索引中的文档 (PG: 触发器自动更新 tsv)"""
        return True
