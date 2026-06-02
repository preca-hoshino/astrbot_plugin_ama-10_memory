"""
官方插件 Page API 适配层。

职责：
1. 为 AstrBot 官方插件页面注册原生 Web API。
2. 直接复用插件运行期组件，不再代理到旧 FastAPI WebUI。
3. 保留返回结构与旧前端尽量一致，降低页面迁移成本。
"""

from __future__ import annotations

import json
import time
from typing import Any

from astrbot.api import logger
from quart import request

from ..storage.pg_connection import get_pool
from .managers.backup_manager import BackupManager

PLUGIN_NAME = "astrbot_plugin_ama10_memory"
PAGE_API_PREFIX = f"/{PLUGIN_NAME}/page"


class PluginPageApi:
    """AMA-10 Memory 插件页面 API。"""

    def __init__(self, plugin) -> None:
        self.plugin = plugin

    def register_routes(self) -> None:
        """注册官方插件页面所需的原生 API。"""
        logger.info("[PageAPI] 开始注册官方插件页面 API...")
        register = self.plugin.context.register_web_api
        api_count = 0
        register(
            f"{PAGE_API_PREFIX}/stats",
            self.get_stats,
            ["GET"],
            "AMA-10 Memory Page stats",
        )
        register(
            f"{PAGE_API_PREFIX}/memories",
            self.list_memories,
            ["GET"],
            "AMA-10 Memory Page memories",
        )
        register(
            f"{PAGE_API_PREFIX}/memories/update",
            self.update_memory,
            ["POST"],
            "AMA-10 Memory Page update memory",
        )
        register(
            f"{PAGE_API_PREFIX}/memories/batch-delete",
            self.batch_delete_memories,
            ["POST"],
            "AMA-10 Memory Page batch delete memories",
        )
        register(
            f"{PAGE_API_PREFIX}/recall/test",
            self.test_recall,
            ["POST"],
            "AMA-10 Memory Page recall test",
        )
        register(
            f"{PAGE_API_PREFIX}/graph/overview",
            self.get_graph_overview,
            ["GET"],
            "AMA-10 Memory Page graph overview",
        )
        register(
            f"{PAGE_API_PREFIX}/graph/query",
            self.query_graph,
            ["POST"],
            "AMA-10 Memory Page graph query",
        )
        register(
            f"{PAGE_API_PREFIX}/backups",
            self.list_backups,
            ["GET"],
            "AMA-10 Memory Page backup list",
        )
        api_count = len([r for r in self.plugin.context.registered_web_apis if r[0].startswith(PAGE_API_PREFIX)])
        logger.info(f"[PageAPI] 已注册 {api_count} 个页面 API (前缀: {PAGE_API_PREFIX})")
        logger.debug(f"[PageAPI] 注册的路由: {[r[0] for r in self.plugin.context.registered_web_apis if r[0].startswith(PAGE_API_PREFIX)]}")

    async def get_stats(self):
        logger.debug("[PageAPI] 收到 GET /stats 请求")
        ready, error = await self._ensure_plugin_ready()
        if error:
            logger.warning(f"[PageAPI] /stats 请求失败: 插件未就绪")
            return error
        del ready

        try:
            stats = await self.plugin.initializer.memory_engine.get_statistics()
            logger.debug(f"[PageAPI] /stats 返回成功: {stats}")
            return self._ok(stats)
        except Exception as exc:
            logger.error(f"[PageAPI] 获取统计信息失败: {exc}", exc_info=True)
            return self._error(str(exc))

    async def list_memories(self):
        ready, error = await self._ensure_plugin_ready()
        if error:
            logger.warning(f"[PageAPI] /memories 请求失败: 插件未就绪")
            return error
        memory_engine = ready["memory_engine"]

        query = request.args
        session_id = str(query.get("session_id", "")).strip() or None
        keyword = str(query.get("keyword", "")).strip()
        status_filter = str(query.get("status", "all")).strip().lower() or "all"

        try:
            page = max(1, int(query.get("page", 1)))
            page_size = min(500, max(1, int(query.get("page_size", 20))))
        except (TypeError, ValueError):
            return self._error("分页参数无效")

        logger.debug(f"[PageAPI] /memories 请求: page={page}, size={page_size}, session={session_id}, keyword={keyword!r}, status={status_filter}")

        offset = (page - 1) * page_size
        where_clauses: list[str] = []
        params: list[Any] = []
        idx = 1

        if session_id:
            where_clauses.append(f"metadata->>'session_id' = ${idx}")
            params.append(session_id)
            idx += 1

        if status_filter != "all":
            where_clauses.append(f"COALESCE(metadata->>'status', 'active') = ${idx}")
            params.append(status_filter)
            idx += 1

        if keyword:
            keyword_like = f"%{keyword}%"
            if keyword.isdigit():
                where_clauses.append(f"(CAST(id AS TEXT) = ${idx} OR text ILIKE ${idx + 1})")
                params.extend([keyword, keyword_like])
                idx += 2
            else:
                where_clauses.append(f"(text ILIKE ${idx} OR COALESCE(metadata->>'memory_type', '') ILIKE ${idx + 1})")
                params.extend([keyword_like, keyword_like])
                idx += 2

        where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        try:
            pool = get_pool()
            async with pool.acquire() as conn:
                count_row = await conn.fetchrow(
                    f"SELECT COUNT(*) AS total FROM documents {where_clause}",
                    *params,
                )
                total = int(count_row["total"]) if count_row else 0

                rows = await conn.fetch(
                    f"""
                    SELECT id, doc_id, text, metadata, created_at, updated_at
                    FROM documents
                    {where_clause}
                    ORDER BY (metadata->>'create_time')::numeric DESC NULLS LAST, id DESC
                    LIMIT ${idx} OFFSET ${idx + 1}
                    """,
                    *params, page_size, offset,
                )
            logger.debug(f"[PageAPI] /memories PG 查询完成: total={total}, 返回 {len(rows)} 行")
        except Exception as exc:
            logger.error(f"[PageAPI] 获取记忆列表失败: {exc}", exc_info=True)
            return self._error(str(exc))

        items: list[dict[str, Any]] = []
        for row in rows:
            meta = row["metadata"]
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (TypeError, json.JSONDecodeError):
                    meta = {}
            items.append(
                {
                    "id": row["id"],
                    "doc_id": row["doc_id"],
                    "text": row["text"],
                    "metadata": self._normalize_metadata(meta),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            )

        return self._ok(
            {
                "items": items,
                "total": total,
                "page": page,
                "page_size": page_size,
                "has_more": (offset + page_size) < total,
            }
        )

    async def update_memory(self):
        ready, error = await self._ensure_plugin_ready()
        if error:
            return error
        memory_engine = ready["memory_engine"]

        payload = await request.get_json(silent=True) or {}
        try:
            memory_id = int(payload.get("memory_id"))
        except (TypeError, ValueError):
            return self._error("memory_id 必须是整数")

        field = str(payload.get("field", "")).strip()
        value = payload.get("value")
        reason = str(payload.get("reason", "")).strip()

        if not field or value is None:
            return self._error("需要指定 field 和 value")

        memory = await self._get_memory_record(memory_id)
        if not memory:
            return self._error("记忆不存在")

        if field == "content":
            new_content = str(value).strip()
            if not new_content:
                return self._error("记忆内容不能为空")

            current_metadata = self._normalize_metadata(memory.get("metadata"))
            session_id = current_metadata.get("session_id")
            persona_id = current_metadata.get("persona_id")
            importance = float(current_metadata.get("importance", 0.5) or 0.5)

            if reason:
                current_metadata["update_reason"] = reason
            current_metadata["updated_at"] = time.time()
            current_metadata["previous_content"] = str(memory.get("text", ""))[:100]

            new_memory_id = None
            try:
                new_memory_id = await memory_engine.add_memory(
                    content=new_content,
                    session_id=session_id,
                    persona_id=persona_id,
                    importance=importance,
                    metadata=current_metadata,
                )
                delete_success = await memory_engine.delete_memory(memory_id)
                if not delete_success:
                    await memory_engine.delete_memory(new_memory_id)
                    return self._error("旧记忆删除失败，已回滚本次内容更新")
            except Exception as exc:
                if new_memory_id is not None:
                    try:
                        await memory_engine.delete_memory(new_memory_id)
                    except Exception:
                        logger.error(
                            f"[PageAPI] 回滚新记忆失败 (new_memory_id={new_memory_id})",
                            exc_info=True,
                        )
                logger.error(f"[PageAPI] 更新记忆内容失败: {exc}", exc_info=True)
                return self._error(str(exc))

            return {
                "status": "ok",
                "data": {
                    "message": f"记忆内容已更新（ID: {memory_id} → {new_memory_id}）",
                    "old_memory_id": memory_id,
                    "new_memory_id": new_memory_id,
                    "field": field,
                },
            }

        updates: dict[str, Any] = {}
        if field == "importance":
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return self._error("重要性必须是数字")
            if 0.0 <= parsed <= 1.0:
                normalized = parsed
            elif 0.0 <= parsed <= 10.0:
                normalized = parsed / 10.0
            else:
                return self._error("重要性必须在 0-1 或 0-10 范围内")
            updates["importance"] = normalized
        elif field == "status":
            status_value = str(value).strip()
            if status_value not in {"active", "archived", "deleted"}:
                return self._error("状态必须是 active、archived 或 deleted")
            updates["metadata"] = {"status": status_value}
        elif field == "type":
            type_value = str(value).strip()
            if not type_value:
                return self._error("类型不能为空")
            updates["metadata"] = {"memory_type": type_value}
        else:
            return self._error(f"不支持编辑字段: {field}")

        if reason:
            updates.setdefault("metadata", {})
            updates["metadata"]["update_reason"] = reason

        try:
            success = await memory_engine.update_memory(memory_id, updates)
        except Exception as exc:
            logger.error(f"[PageAPI] 更新记忆失败: {exc}", exc_info=True)
            return self._error(str(exc))

        if not success:
            return self._error("更新失败")

        return {
            "status": "ok",
            "data": {
                "message": f"记忆 {memory_id} 的 {field} 已更新",
                "memory_id": memory_id,
                "field": field,
            },
        }

    async def batch_delete_memories(self):
        ready, error = await self._ensure_plugin_ready()
        if error:
            return error
        memory_engine = ready["memory_engine"]

        payload = await request.get_json(silent=True) or {}
        memory_ids = payload.get("memory_ids", [])
        if not isinstance(memory_ids, list) or not memory_ids:
            return self._error("需要提供记忆 ID 列表")

        deleted_count = 0
        failed_count = 0
        failed_ids: list[Any] = []

        valid_ids: list[int] = []
        for raw_id in memory_ids:
            try:
                valid_ids.append(int(raw_id))
            except Exception:
                failed_count += 1
                failed_ids.append(raw_id)

        if valid_ids:
            deleted_count = await memory_engine.batch_delete_memories(valid_ids)

        return self._ok(
            {
                "deleted_count": deleted_count,
                "failed_count": failed_count,
                "total": len(memory_ids),
                "failed_ids": failed_ids,
            }
        )

    async def test_recall(self):
        logger.info("[PageAPI] 收到 POST /recall/test 请求")
        ready, error = await self._ensure_plugin_ready()
        if error:
            logger.warning(f"[PageAPI] /recall/test 失败: 插件未就绪")
            return error
        memory_engine = ready["memory_engine"]

        payload = await request.get_json(silent=True) or {}
        query_text = str(payload.get("query", "")).strip()
        logger.debug(f"[PageAPI] /recall/test payload: query={query_text!r}, k={payload.get('k')}, session_id={payload.get('session_id')}")
        if not query_text:
            logger.warning("[PageAPI] /recall/test 失败: 查询内容为空")
            return self._error("查询内容不能为空")

        try:
            k = min(50, max(1, int(payload.get("k", 5))))
        except (TypeError, ValueError):
            return self._error("k 必须是整数")

        session_id = payload.get("session_id")

        try:
            start_time = time.time()
            logger.info(f"[PageAPI] /recall/test 开始搜索: query={query_text!r}, k={k}, session={session_id}")
            results = await memory_engine.search_memories(
                query=query_text,
                k=k,
                session_id=session_id,
                persona_id=None,
            )
            elapsed_time = (time.time() - start_time) * 1000
            logger.info(f"[PageAPI] /recall/test 搜索完成: 返回 {len(results)} 条, 耗时 {elapsed_time:.1f}ms")
        except Exception as exc:
            logger.error(f"[PageAPI] 召回测试异常: {type(exc).__name__}: {exc}", exc_info=True)
            return self._error(f"召回异常: {type(exc).__name__}: {exc}")

        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "memory_id": result.doc_id,
                    "content": result.content,
                    "similarity_score": round(float(result.final_score), 4),
                    "score_percentage": round(float(result.final_score) * 100, 2),
                    "metadata": {
                        "session_id": result.metadata.get("session_id"),
                        "persona_id": result.metadata.get("persona_id"),
                        "importance": result.metadata.get("importance", 0.5),
                        "memory_type": result.metadata.get("memory_type", "GENERAL"),
                        "status": result.metadata.get("status", "active"),
                        "create_time": result.metadata.get("create_time"),
                    },
                }
            )

        return self._ok(
            {
                "results": formatted_results,
                "total": len(formatted_results),
                "query": query_text,
                "k": k,
                "session_id_filter": session_id,
                "elapsed_time_ms": round(elapsed_time, 2),
            }
        )

    async def get_graph_overview(self):
        ready, error = await self._ensure_plugin_ready()
        if error:
            return error
        memory_engine = ready["memory_engine"]

        args = request.args
        session_id = str(args.get("session_id", "")).strip() or None
        persona_id = str(args.get("persona_id", "")).strip() or None

        try:
            limit_memories = max(1, min(int(args.get("limit_memories", 12)), 24))
            limit_entries = max(12, min(int(args.get("limit_entries", 36)), 80))
            limit_nodes = max(12, min(int(args.get("limit_nodes", 48)), 80))
            limit_edges = max(12, min(int(args.get("limit_edges", 72)), 120))
        except (TypeError, ValueError):
            return self._error("图谱分页参数无效")

        try:
            stats = await memory_engine.get_statistics()
            graph_store = self._get_graph_store(memory_engine)
            empty_snapshot = {
                "nodes": [],
                "edges": [],
                "entries": [],
                "memories": [],
            }
            if graph_store is None:
                return self._ok(
                    self._build_graph_view_payload(
                        empty_snapshot,
                        stats,
                        enabled=False,
                        mode="overview",
                        filters={
                            "session_id": session_id,
                            "persona_id": persona_id,
                        },
                    )
                )

            snapshot = await graph_store.get_graph_snapshot(
                session_id=session_id,
                persona_id=persona_id,
                limit_memories=limit_memories,
                limit_entries=limit_entries,
                limit_nodes=limit_nodes,
                limit_edges=limit_edges,
            )
            return self._ok(
                self._build_graph_view_payload(
                    snapshot,
                    stats,
                    enabled=True,
                    mode="overview",
                    filters={
                        "session_id": session_id,
                        "persona_id": persona_id,
                    },
                )
            )
        except Exception as exc:
            logger.error(f"[PageAPI] 获取图谱概览失败: {exc}", exc_info=True)
            return self._error(str(exc))

    async def query_graph(self):
        ready, error = await self._ensure_plugin_ready()
        if error:
            return error
        memory_engine = ready["memory_engine"]

        payload = await request.get_json(silent=True) or {}
        query_text = str(payload.get("query", "")).strip()
        session_id = str(payload.get("session_id", "")).strip() or None
        persona_id = str(payload.get("persona_id", "")).strip() or None
        memory_id_raw = payload.get("memory_id")

        try:
            limit_memories = max(1, min(int(payload.get("limit_memories", 10)), 24))
            limit_entries = max(12, min(int(payload.get("limit_entries", 40)), 80))
            limit_nodes = max(12, min(int(payload.get("limit_nodes", 56)), 80))
            limit_edges = max(12, min(int(payload.get("limit_edges", 96)), 120))
        except (TypeError, ValueError):
            return self._error("图谱检索参数无效")

        try:
            stats = await memory_engine.get_statistics()
            graph_store = self._get_graph_store(memory_engine)
            empty_snapshot = {
                "nodes": [],
                "edges": [],
                "entries": [],
                "memories": [],
            }
            if graph_store is None:
                return self._ok(
                    self._build_graph_view_payload(
                        empty_snapshot,
                        stats,
                        enabled=False,
                        mode="query",
                        query=query_text,
                        filters={
                            "session_id": session_id,
                            "persona_id": persona_id,
                        },
                    )
                )

            if memory_id_raw not in (None, ""):
                try:
                    memory_id = int(memory_id_raw)
                except (TypeError, ValueError):
                    return self._error("memory_id 必须是整数")

                snapshot = await graph_store.get_subgraph_for_memories(
                    [memory_id],
                    limit_entries=limit_entries,
                    limit_nodes=limit_nodes,
                    limit_edges=limit_edges,
                )
                return self._ok(
                    self._build_graph_view_payload(
                        snapshot,
                        stats,
                        enabled=True,
                        mode="memory_focus",
                        memory_id=memory_id,
                        filters={
                            "session_id": session_id,
                            "persona_id": persona_id,
                        },
                    )
                )

            if not query_text:
                snapshot = await graph_store.get_graph_snapshot(
                    session_id=session_id,
                    persona_id=persona_id,
                    limit_memories=limit_memories,
                    limit_entries=limit_entries,
                    limit_nodes=limit_nodes,
                    limit_edges=limit_edges,
                )
                return self._ok(
                    self._build_graph_view_payload(
                        snapshot,
                        stats,
                        enabled=True,
                        mode="overview",
                        filters={
                            "session_id": session_id,
                            "persona_id": persona_id,
                        },
                    )
                )

            search_results = await memory_engine.search_memories(
                query=query_text,
                k=limit_memories,
                session_id=session_id,
                persona_id=persona_id,
            )
            retrieval_items = []
            matched_memory_ids: list[int] = []
            seen_memory_ids: set[int] = set()
            for result in search_results:
                memory_id = int(result.doc_id)
                if memory_id not in seen_memory_ids:
                    seen_memory_ids.add(memory_id)
                    matched_memory_ids.append(memory_id)
                retrieval_items.append(
                    {
                        "memory_id": memory_id,
                        "content": result.content,
                        "metadata": result.metadata,
                        "final_score": round(float(result.final_score), 6),
                        "rrf_score": round(float(result.rrf_score), 6),
                        "bm25_score": (
                            round(float(result.bm25_score), 6)
                            if result.bm25_score is not None
                            else None
                        ),
                        "vector_score": (
                            round(float(result.vector_score), 6)
                            if result.vector_score is not None
                            else None
                        ),
                        "score_breakdown": {
                            key: round(float(value), 6)
                            for key, value in (result.score_breakdown or {}).items()
                        },
                    }
                )

            tokens = self._tokenize_graph_query(query_text)
            matched_node_ids: list[int] = []
            if tokens:
                node_hits = await graph_store.search_nodes_by_tokens(
                    tokens,
                    limit=max(8, min(limit_nodes, 24)),
                )
                matched_node_ids = [int(item["id"]) for item in node_hits]

                node_entry_hits = await graph_store.get_entries_for_node_ids(
                    matched_node_ids,
                    limit=max(8, min(limit_entries, 24)),
                    session_id=session_id,
                    persona_id=persona_id,
                )
                for hit in node_entry_hits:
                    memory_id = int(hit["source_memory_id"])
                    if memory_id not in seen_memory_ids:
                        seen_memory_ids.add(memory_id)
                        matched_memory_ids.append(memory_id)

            snapshot = await graph_store.get_subgraph_for_memories(
                matched_memory_ids[:limit_memories],
                limit_entries=limit_entries,
                limit_nodes=limit_nodes,
                limit_edges=limit_edges,
            )
            return self._ok(
                self._build_graph_view_payload(
                    snapshot,
                    stats,
                    enabled=True,
                    mode="query",
                    query=query_text,
                    retrieval_items=retrieval_items,
                    matched_node_ids=matched_node_ids,
                    filters={
                        "session_id": session_id,
                        "persona_id": persona_id,
                    },
                )
            )
        except Exception as exc:
            logger.error(f"[PageAPI] 图谱查询失败: {exc}", exc_info=True)
            return self._error(str(exc))

    async def _ensure_plugin_ready(self) -> tuple[dict[str, Any] | None, dict | None]:
        ready, message = await self.plugin._ensure_plugin_ready()
        if not ready:
            logger.warning(f"[PageAPI] 插件未就绪: {message}")
            return None, self._error(message or "插件尚未就绪")

        memory_engine = self.plugin.initializer.memory_engine
        if memory_engine is None:
            logger.error("[PageAPI] memory_engine 为 None")
            return None, self._error("记忆引擎未初始化")

        logger.debug("[PageAPI] 插件就绪检查通过")
        return {
            "memory_engine": memory_engine,
            "conversation_manager": self.plugin.initializer.conversation_manager,
        }, None

    async def _get_memory_record(self, memory_id: int) -> dict[str, Any] | None:
        memory_engine = self.plugin.initializer.memory_engine
        if memory_engine is None:
            return None

        memory = await memory_engine.get_memory(memory_id)
        if memory:
            return memory

        try:
            pool = get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT id, text, metadata FROM documents WHERE id = $1",
                    memory_id,
                )
        except Exception:
            return None

        if not row:
            return None

        return {
            "id": row["id"],
            "text": row["text"],
            "metadata": self._normalize_metadata(row["metadata"]),
        }

    @staticmethod
    def _normalize_metadata(metadata: Any) -> dict[str, Any]:
        if isinstance(metadata, dict):
            return metadata
        if not metadata:
            return {}
        try:
            parsed = json.loads(metadata)
        except (TypeError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _ok(data: Any = None) -> dict[str, Any]:
        return {"success": True, "data": data, "status": "ok"}

    @staticmethod
    def _error(message: str) -> dict[str, Any]:
        return {"success": False, "error": str(message), "status": "error", "message": str(message)}

    @staticmethod
    def _get_graph_store(memory_engine):
        return getattr(memory_engine, "graph_store", None)

    @staticmethod
    def _tokenize_graph_query(query: str) -> list[str]:
        query_text = str(query or "").strip().lower()
        if not query_text:
            return []

        normalized = "".join(
            character if character.isalnum() else " " for character in query_text
        )
        raw_tokens = [token for token in normalized.split() if token]
        tokens: list[str] = []
        seen: set[str] = set()

        def add_token(value: str):
            token = value.strip()
            if len(token) < 2 or token in seen:
                return
            seen.add(token)
            tokens.append(token)

        for token in raw_tokens:
            add_token(token)

        compact = "".join(character for character in query_text if character.isalnum())
        if compact and any(ord(character) > 127 for character in compact):
            add_token(compact)
            for size in (2, 3):
                if len(tokens) >= 12:
                    break
                max_index = max(0, len(compact) - size + 1)
                for index in range(max_index):
                    add_token(compact[index : index + size])
                    if len(tokens) >= 12:
                        break

        return tokens[:12]

    @staticmethod
    def _build_graph_view_payload(
        snapshot: dict[str, Any],
        stats: dict[str, Any],
        *,
        enabled: bool,
        mode: str,
        query: str | None = None,
        memory_id: int | None = None,
        retrieval_items: list[dict[str, Any]] | None = None,
        matched_node_ids: list[int] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        nodes = [dict(item) for item in snapshot.get("nodes", [])]
        edges = [dict(item) for item in snapshot.get("edges", [])]
        entries = [dict(item) for item in snapshot.get("entries", [])]
        memories = [dict(item) for item in snapshot.get("memories", [])]
        retrieval_items = [dict(item) for item in (retrieval_items or [])]
        matched_node_ids = [int(item) for item in (matched_node_ids or [])]
        matched_node_id_set = set(matched_node_ids)
        retrieval_lookup = {
            int(item["memory_id"]): item
            for item in retrieval_items
            if item.get("memory_id") is not None
        }

        node_type_breakdown: dict[str, int] = {}
        relation_breakdown: dict[str, int] = {}

        for node in nodes:
            node["highlighted"] = int(node.get("id", 0)) in matched_node_id_set
            node_type = str(node.get("type", "unknown") or "unknown")
            node_type_breakdown[node_type] = node_type_breakdown.get(node_type, 0) + 1

        for edge in edges:
            relation_type = str(edge.get("relation_type", "related") or "related")
            relation_breakdown[relation_type] = (
                relation_breakdown.get(relation_type, 0) + 1
            )

        for memory in memories:
            memory_key = memory.get("memory_id")
            if memory_key is None:
                continue
            retrieval = retrieval_lookup.get(int(memory_key))
            if retrieval is not None:
                memory["retrieval"] = retrieval

        top_nodes = sorted(
            nodes,
            key=lambda item: (
                -float(item.get("weight", 0.0)),
                -int(item.get("degree", 0)),
                str(item.get("label", "")),
            ),
        )[:8]
        top_memories = sorted(
            memories,
            key=lambda item: (
                -float((item.get("retrieval") or {}).get("final_score", -1.0)),
                -int(item.get("entry_count", 0)),
                -int(item.get("node_count", 0)),
                -int(item.get("edge_count", 0)),
                -float(item.get("importance", 0.0)),
            ),
        )[:8]

        summary = {
            "visible_node_count": len(nodes),
            "visible_edge_count": len(edges),
            "visible_entry_count": len(entries),
            "visible_memory_count": len(memories),
            "graph_node_count": int(stats.get("graph_nodes", 0) or 0),
            "graph_edge_count": int(stats.get("graph_edges", 0) or 0),
            "graph_entry_count": int(stats.get("graph_entries", 0) or 0),
            "graph_memory_enabled": bool(enabled),
            "node_type_breakdown": node_type_breakdown,
            "relation_breakdown": relation_breakdown,
        }

        return {
            "enabled": enabled,
            "mode": mode,
            "query": query or None,
            "memory_id": memory_id,
            "filters": filters or {},
            "summary": summary,
            "matched_node_ids": matched_node_ids,
            "matched_memory_ids": [item["memory_id"] for item in retrieval_items],
            "top_nodes": top_nodes,
            "top_memories": top_memories,
            "retrieval": {
                "total": len(retrieval_items),
                "items": retrieval_items,
            },
            "snapshot": {
                "nodes": nodes,
                "edges": edges,
                "entries": entries,
                "memories": memories,
            },
        }

    async def list_backups(self):
        """列出所有版本备份及其元数据。"""
        data_dir = self.plugin.initializer.data_dir if self.plugin.initializer else ""
        if not data_dir:
            return self._ok({"backups": [], "total": 0})
        backups = BackupManager.list_backups(data_dir)
        return self._ok({"backups": backups, "total": len(backups)})
