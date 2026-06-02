"""
PgVecDB — 基于 pgvector 的 BaseVecDB 实现
替代 FaissVecDB，将向量存储到 PostgreSQL。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import numpy as np

from astrbot.api import logger
from astrbot.core.db.vec_db.base import BaseVecDB, Result

from .pg_connection import get_pool


class PgVecDB(BaseVecDB):
    """PostgreSQL + pgvector 向量数据库实现"""

    # ---- 模块级共享 embedding 缓存（跨实例共享，避免同次召回重复 API 调用）----
    _shared_cache_query: str | None = None
    _shared_cache_embedding: list[float] | None = None
    _shared_cache_ts: float = 0.0
    _shared_cache_lock: asyncio.Lock = asyncio.Lock()
    _CACHE_TTL: float = 30.0  # 秒

    def __init__(
        self,
        vec_table: str = "documents_vec",
        doc_table: str = "documents",
        dimension: int = 1024,
        embedding_provider=None,
        provider_getter=None,
    ):
        self.vec_table = vec_table
        self.doc_table = doc_table
        self.dimension = dimension
        # provider_getter: 可调用对象，返回当前活跃的 embedding provider
        # 用于避免持有 stale 引用（AstrBot 可能 terminate 旧 provider 并创建新实例）
        self._provider_getter = provider_getter
        self.embedding_provider = embedding_provider

    @property
    def _current_provider(self):
        """动态获取当前活跃的 embedding provider"""
        if self._provider_getter is not None:
            try:
                provider = self._provider_getter()
                if provider is not None:
                    self.embedding_provider = provider
            except Exception:
                pass
        return self.embedding_provider

    async def initialize(self) -> None:
        pass  # 表结构由迁移脚本创建

    async def insert(
        self,
        content: str,
        metadata: dict | None = None,
        id: str | None = None,
    ) -> int:
        pool = get_pool()
        str_id = id or str(uuid.uuid4())
        metadata = metadata or {}
        logger.debug(f"[PgVecDB] insert: doc_id={str_id}, content_len={len(content)}")

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"INSERT INTO {self.doc_table} (doc_id, text, metadata) "
                "VALUES ($1, $2, $3::jsonb) RETURNING id",
                str_id,
                content,
                json.dumps(metadata, ensure_ascii=False),
            )
            int_id = row["id"]

            vec = await self._get_embedding(content)
            await conn.execute(
                f"INSERT INTO {self.vec_table} (doc_id, embedding) "
                "VALUES ($1, $2::vector)",
                int_id,
                self._vec_to_str(vec),
            )
            return int_id

    async def insert_batch(
        self,
        contents: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        batch_size: int = 32,
        tasks_limit: int = 3,
        max_retries: int = 3,
        progress_callback=None,
    ) -> list[int]:
        if not contents:
            return []

        metadatas = metadatas or [{} for _ in contents]
        ids = ids or [str(uuid.uuid4()) for _ in contents]
        pool = get_pool()
        int_ids: list[int] = []

        async with pool.acquire() as conn:
            async with conn.transaction():
                for i, (content, meta, str_id) in enumerate(zip(contents, metadatas, ids)):
                    row = await conn.fetchrow(
                        f"INSERT INTO {self.doc_table} (doc_id, text, metadata) "
                        "VALUES ($1, $2, $3::jsonb) RETURNING id",
                        str_id,
                        content,
                        json.dumps(meta, ensure_ascii=False),
                    )
                    int_ids.append(row["id"])
                    if progress_callback:
                        progress_callback(i + 1, len(contents))

        return int_ids

    async def insert_vectors_batch(
        self,
        int_ids: list[int],
        vectors: list[list[float]],
        batch_size: int = 100,
    ) -> None:
        """批量插入向量（用于索引重建）"""
        pool = get_pool()
        async with pool.acquire() as conn:
            for i in range(0, len(int_ids), batch_size):
                batch_ids = int_ids[i : i + batch_size]
                batch_vecs = vectors[i : i + batch_size]
                values_parts = []
                params = []
                for j, (iid, vec) in enumerate(zip(batch_ids, batch_vecs)):
                    values_parts.append(f"(${j * 2 + 1}, ${j * 2 + 2}::vector)")
                    params.append(iid)
                    params.append(self._vec_to_str(vec))
                sql = (
                    f"INSERT INTO {self.vec_table} (doc_id, embedding) "
                    f"VALUES {','.join(values_parts)} ON CONFLICT (doc_id) DO NOTHING"
                )
                await conn.execute(sql, *params)

    # ---- 类级共享 embedding 缓存（跨所有 PgVecDB 实例共享，避免同次召回重复 API 调用）----
    _shared_cache_query: str | None = None
    _shared_cache_embedding: list[float] | None = None
    _shared_cache_ts: float = 0.0
    _shared_cache_lock: asyncio.Lock | None = None  # 延迟初始化，避免模块加载时无事件循环
    _CACHE_TTL: float = 30.0  # 秒

    @classmethod
    def _ensure_lock(cls) -> asyncio.Lock:
        """延迟初始化共享锁，确保在事件循环运行后创建。"""
        if cls._shared_cache_lock is None:
            cls._shared_cache_lock = asyncio.Lock()
        return cls._shared_cache_lock

    async def _resolve_embedding(self, query: str, query_embedding: list[float] | None = None) -> list[float]:
        """返回 query 的 embedding，优先使用预计算值或共享缓存。

        使用类级变量 + asyncio.Lock 保证：
        - 同一查询只调用一次 embedding API
        - 并发协程等待而非重复请求
        """
        import time as _time
        if query_embedding is not None:
            return query_embedding
        now = _time.time()
        # 快速路径：无锁读
        if (
            PgVecDB._shared_cache_query == query
            and PgVecDB._shared_cache_embedding is not None
            and (now - PgVecDB._shared_cache_ts) < PgVecDB._CACHE_TTL
        ):
            logger.debug("[PgVecDB] embedding 命中共享缓存")
            return PgVecDB._shared_cache_embedding
        # 慢路径：加锁 + double-check
        async with PgVecDB._ensure_lock():
            now2 = _time.time()
            if (
                PgVecDB._shared_cache_query == query
                and PgVecDB._shared_cache_embedding is not None
                and (now2 - PgVecDB._shared_cache_ts) < PgVecDB._CACHE_TTL
            ):
                logger.debug("[PgVecDB] embedding 命中共享缓存 (double-check)")
                return PgVecDB._shared_cache_embedding
            vec = await self._get_embedding(query)
            PgVecDB._shared_cache_query = query
            PgVecDB._shared_cache_embedding = vec
            PgVecDB._shared_cache_ts = _time.time()
            return vec

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 20,
        rerank: bool = False,
        metadata_filters: dict | None = None,
        query_embedding: list[float] | None = None,
        **kwargs,
    ) -> list[Result]:
        # 兼容调用方传 k=top_k 的情况
        if "k" in kwargs:
            top_k = kwargs.pop("k")
        pool = get_pool()
        logger.debug(f"[PgVecDB] retrieve: query={query[:50]!r}, top_k={top_k}, fetch_k={fetch_k}, filters={metadata_filters}")
        vec = await self._resolve_embedding(query, query_embedding)
        logger.debug(f"[PgVecDB] retrieve: embedding 维度={len(vec)}")
        vec_str = self._vec_to_str(vec)

        where_clauses = []
        params: list[Any] = [vec_str, fetch_k]
        idx = 3

        if metadata_filters:
            for key, value in metadata_filters.items():
                where_clauses.append(f"d.metadata->>'{key}' = ${idx}")
                params.append(str(value))
                idx += 1

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        sql = f"""
            SELECT d.id, d.doc_id, d.text, d.metadata,
                   1 - (v.embedding <=> $1::vector) AS similarity
            FROM {self.vec_table} v
            JOIN {self.doc_table} d ON d.id = v.doc_id
            {where_sql}
            ORDER BY v.embedding <=> $1::vector
            LIMIT $2
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        logger.debug(f"[PgVecDB] retrieve 完成: 返回 {len(rows)} 条结果 (top_k={top_k})")
        results = []
        for row in rows:
            metadata = row["metadata"]
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
            results.append(
                Result(
                    similarity=float(row["similarity"]),
                    data={
                        "id": row["id"],
                        "doc_id": row["doc_id"],
                        "text": row["text"],
                        "metadata": metadata or {},
                        **(metadata or {}),
                    },
                )
            )

        return results[:top_k]

    async def delete(self, doc_id: str) -> bool:
        pool = get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT id FROM {self.doc_table} WHERE doc_id = $1", doc_id
            )
            if not row:
                return False
            int_id = row["id"]
            await conn.execute(
                f"DELETE FROM {self.vec_table} WHERE doc_id = $1", int_id
            )
            await conn.execute(
                f"DELETE FROM {self.doc_table} WHERE id = $1", int_id
            )
            return True

    async def close(self) -> None:
        pass  # 连接池由 pg_connection 管理

    async def count_documents(self, metadata_filters: dict | None = None, **kwargs) -> int:
        # 兼容 metadata_filter (单数) 和 metadata_filters (复数)
        filters = metadata_filters or kwargs.get("metadata_filter")
        pool = get_pool()
        async with pool.acquire() as conn:
            if filters:
                conditions = []
                params = []
                for i, (k, v) in enumerate(filters.items(), 1):
                    conditions.append(f"metadata->>'{k}' = ${i}")
                    params.append(str(v))
                where = f"WHERE {' AND '.join(conditions)}"
                row = await conn.fetchrow(
                    f"SELECT COUNT(*) AS cnt FROM {self.doc_table} {where}", *params
                )
            else:
                row = await conn.fetchrow(
                    f"SELECT COUNT(*) AS cnt FROM {self.doc_table}"
                )
            return row["cnt"]

    async def delete_documents(self, metadata_filters: dict) -> None:
        pool = get_pool()
        conditions = []
        params = []
        for i, (k, v) in enumerate(metadata_filters.items(), 1):
            conditions.append(f"metadata->>'{k}' = ${i}")
            params.append(str(v))
        where = f"WHERE {' AND '.join(conditions)}"
        async with pool.acquire() as conn:
            ids = await conn.fetch(
                f"SELECT id FROM {self.doc_table} {where}", *params
            )
            id_list = [r["id"] for r in ids]
            if id_list:
                id_placeholders = ",".join(f"${i + 1}" for i in range(len(id_list)))
                await conn.execute(
                    f"DELETE FROM {self.vec_table} WHERE doc_id IN ({id_placeholders})",
                    *id_list,
                )
                await conn.execute(
                    f"DELETE FROM {self.doc_table} WHERE id IN ({id_placeholders})",
                    *id_list,
                )

    async def get_document_by_doc_id(self, doc_id: str) -> dict | None:
        pool = get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT id, doc_id, text, metadata FROM {self.doc_table} WHERE doc_id = $1",
                doc_id,
            )
            if not row:
                return None
            meta = row["metadata"]
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}
            return {
                "id": row["id"],
                "doc_id": row["doc_id"],
                "text": row["text"],
                **(meta or {}),
            }

    @property
    def document_storage(self):
        """兼容 FaissVecDB 的 document_storage 访问模式"""
        return self

    async def get_documents(
        self,
        metadata_filters: dict,
        ids: list | None = None,
        offset: int | None = 0,
        limit: int | None = 100,
    ) -> list[dict]:
        """兼容 DocumentStorage.get_documents 接口"""
        pool = get_pool()
        conditions = []
        params: list = []
        idx = 1

        for key, val in metadata_filters.items():
            conditions.append(f"metadata->>'{key}' = ${idx}")
            params.append(str(val))
            idx += 1

        if ids is not None:
            valid_ids = [int(i) for i in ids if i != -1]
            if valid_ids:
                placeholders = ",".join(f"${idx + i}" for i in range(len(valid_ids)))
                conditions.append(f"id IN ({placeholders})")
                params.extend(valid_ids)
                idx += len(valid_ids)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = f"LIMIT ${idx}" if limit is not None else ""
        if limit is not None:
            params.append(limit)
            idx += 1
        offset_clause = f"OFFSET ${idx}" if offset and offset > 0 else ""
        if offset and offset > 0:
            params.append(offset)

        sql = (
            f"SELECT id, doc_id, text, metadata, created_at, updated_at "
            f"FROM {self.doc_table} {where} ORDER BY id {limit_clause} {offset_clause}"
        )

        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        results = []
        for row in rows:
            meta = row["metadata"]
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}
            results.append({
                "id": row["id"],
                "doc_id": row["doc_id"],
                "text": row["text"],
                "metadata": meta or {},
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            })
        return results

    async def get_document_by_id(self, doc_id: int) -> dict | None:
        """根据整数 ID 获取单个文档"""
        pool = get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT id, doc_id, text, metadata FROM {self.doc_table} WHERE id = $1",
                doc_id,
            )
        if not row:
            return None
        meta = row["metadata"]
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        return {
            "id": row["id"],
            "doc_id": row["doc_id"],
            "text": row["text"],
            "metadata": meta or {},
        }

    async def update_document_metadata(self, doc_id: int, metadata: dict) -> bool:
        """更新文档 metadata (PG 模式专用，替代 SQLAlchemy session)"""
        pool = get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                f"UPDATE {self.doc_table} SET metadata = $1::jsonb WHERE id = $2",
                json.dumps(metadata, ensure_ascii=False),
                doc_id,
            )
            # asyncpg 返回 "UPDATE N"，N > 0 表示有行被更新
            try:
                return int(result.split()[-1]) > 0
            except (ValueError, IndexError):
                return False

    class _PgSession:
        """模拟 SQLAlchemy session 接口，用于 vector_retriever.update_metadata()"""

        def __init__(self, pool):
            self._pool = pool
            self._conn = None

        async def __aenter__(self):
            self._conn = await self._pool.acquire()
            return self

        async def __aexit__(self, *args):
            if self._conn:
                await self._pool.release(self._conn)
            return False

        def begin(self):
            return self._PgTransaction(self._conn)

        async def execute(self, stmt, params=None):
            sql = str(stmt)
            if params:
                meta = params.get("metadata", "")
                doc_id = params.get("id", 0)
                await self._conn.execute(
                    f"UPDATE documents SET metadata = $1::jsonb WHERE id = $2",
                    str(meta), int(doc_id),
                )

        class _PgTransaction:
            def __init__(self, conn):
                self._conn = conn
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False

    def get_session(self):
        """兼容 DocumentStorage.get_session() 接口"""
        return self._PgSession(get_pool())

    # 模式检测: -1=未检测, 0=正常模式, 1=fallback 模式(不传 dimensions)
    _embed_mode: int = -1

    @staticmethod
    def _is_client_closed(e: Exception) -> bool:
        """检测是否为 httpx 客户端已关闭的错误 (含异常链)"""
        def _check(exc) -> bool:
            msg = str(exc).lower()
            return "client has been closed" in msg or "cannot send a request" in msg
        # 检查当前异常及完整 __cause__ 链
        cur = e
        while cur is not None:
            if _check(cur):
                return True
            cur = getattr(cur, "__cause__", None)
        return False

    @staticmethod
    def _recreate_client(provider) -> bool:
        """尝试重建 embedding provider 的 AsyncOpenAI 客户端"""
        try:
            import httpx as _httpx
            from openai import AsyncOpenAI

            old_client = getattr(provider, "client", None)
            if old_client is None:
                return False

            # 从旧客户端提取配置
            base_url = str(old_client.base_url)
            api_key = old_client.api_key
            timeout_val = getattr(old_client, "timeout", 20)

            # proxy: 已关闭的客户端无法提取 proxy 配置，重建时不保留
            http_client = None

            new_client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout_val,
                http_client=http_client,
            )
            provider.client = new_client
            logger.info("[PgVecDB] 已重建 embedding 客户端 (检测到旧连接已关闭)")
            return True
        except Exception as rebuild_err:
            logger.error(f"[PgVecDB] 重建 embedding 客户端失败: {rebuild_err}")
            return False

    async def _get_embedding(self, text: str) -> list[float]:
        """获取文本的 embedding 向量。

        默认使用不带 dimensions 参数的 fallback 模式（兼容 SiliconFlow 等 API）。
        仅当 fallback 也失败时，才尝试标准模式。
        """
        provider = self._current_provider
        if provider is None:
            raise RuntimeError(
                "PgVecDB: embedding_provider 未设置，无法生成向量。"
                "请确保 Embedding Provider 已正确配置。"
            )

        try:
            # 默认 fallback: 不传 dimensions 参数，兼容性最好
            result = await self._get_embedding_fallback(text)
            logger.debug(f"[PgVecDB] embedding 成功: dim={len(result)}, text={text[:30]!r}")
            return result
        except Exception as e:
            # 客户端被关闭: 尝试重建并重试一次
            if self._is_client_closed(e):
                logger.warning("[PgVecDB] embedding 客户端已关闭，尝试重建...")
                if self._provider_getter is not None:
                    try:
                        new_provider = self._provider_getter()
                        if new_provider is not None and new_provider is not provider:
                            self.embedding_provider = new_provider
                            logger.info("[PgVecDB] 已从 AstrBot 获取新的 embedding provider")
                            return await self._get_embedding_fallback(text)
                    except Exception:
                        pass
                if self._recreate_client(provider):
                    return await self._get_embedding_fallback(text)
                else:
                    raise RuntimeError("PgVecDB: embedding 客户端已关闭且无法重建") from e
            # fallback 失败，尝试标准模式（带 dimensions）
            try:
                result = await provider.get_embedding(text)
                return result
            except Exception:
                raise e from e

    async def _get_embedding_fallback(self, text: str) -> list[float]:
        """不带 dimensions 参数的 embedding 调用 (兼容 SiliconFlow 等不支持该参数的 API)"""
        provider = self._current_provider
        client = getattr(provider, "client", None)
        model = getattr(provider, "model", None)
        if client is None or model is None:
            raise RuntimeError(
                "PgVecDB: 无法访问底层 embedding client，请检查 Embedding Provider 配置。"
            )
        # 检测客户端是否已关闭
        httpx_client = getattr(client, "_client", None)
        if httpx_client and getattr(httpx_client, "is_closed", False):
            logger.warning("[PgVecDB] fallback 模式: embedding 客户端已关闭，尝试重建...")
            if self._recreate_client(provider):
                client = provider.client
            else:
                raise RuntimeError("PgVecDB: embedding 客户端已关闭且无法重建")
        response = await client.embeddings.create(input=text, model=model)
        logger.debug(f"[PgVecDB] _get_embedding_fallback 成功: model={model}, dim={len(response.data[0].embedding)}")
        return response.data[0].embedding

    @staticmethod
    def _vec_to_str(vec: list[float]) -> str:
        return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
