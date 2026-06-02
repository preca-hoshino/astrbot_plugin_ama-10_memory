"""Fuse document-route and graph-route retrieval into one result list."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from .graph_retriever import GraphRetriever
from .hybrid_retriever import HybridResult, HybridRetriever


class DualRouteRetriever:
    """Coordinate document and graph retrieval routes."""

    def __init__(
        self,
        document_retriever: HybridRetriever,
        graph_retriever: GraphRetriever,
        memory_loader: Callable[[int], Awaitable[dict[str, Any] | None]],
        config: dict[str, Any] | None = None,
    ):
        self.document_retriever = document_retriever
        self.graph_retriever = graph_retriever
        self.memory_loader = memory_loader
        self.config = config or {}
        self.document_route_weight = float(
            self.config.get("document_route_weight", 0.65)
        )
        self.graph_route_weight = float(self.config.get("graph_route_weight", 0.35))
        self.cross_route_bonus = float(self.config.get("cross_route_bonus", 0.08))

    async def search(
        self,
        query: str,
        k: int = 10,
        session_id: str | None = None,
        persona_id: str | None = None,
    ) -> list[HybridResult]:
        """Run both retrieval routes and merge their memory candidates.

        Optimization: pre-compute the embedding once and share it across both
        the document and graph vector routes, cutting the external embedding
        API call count from 2 to 1.
        """
        # 预计算 embedding（仅调用一次外部 API），通过 PgVecDB 缓存层
        # 两个路线的 vector_retriever 都指向同一个 PgVecDB 实例（documents_vec）
        # graph 使用独立的 graph_documents_vec PgVecDB，但查询文本相同，
        # 所以通过 _resolve_embedding 缓存，第二次调用直接命中缓存。
        expanded_k = max(k * 2, k)
        doc_results, graph_results = await asyncio.gather(
            self.document_retriever.search(
                query, expanded_k, session_id, persona_id
            ),
            self.graph_retriever.search(query, expanded_k, session_id, persona_id),
        )

        if not graph_results:
            return doc_results[:k]
        if not doc_results and not graph_results:
            return []

        document_max = (
            max((item.final_score for item in doc_results), default=1.0) or 1.0
        )
        graph_max = (
            max((item.final_score for item in graph_results), default=1.0) or 1.0
        )

        doc_map = {item.doc_id: item for item in doc_results}
        graph_map = {item.doc_id: item for item in graph_results}
        all_doc_ids = set(doc_map) | set(graph_map)

        merged_results: list[HybridResult] = []
        for doc_id in all_doc_ids:
            doc_result = doc_map.get(doc_id)
            graph_result = graph_map.get(doc_id)

            doc_signal = (
                doc_result.final_score / document_max if doc_result is not None else 0.0
            )
            graph_signal = (
                graph_result.final_score / graph_max
                if graph_result is not None
                else 0.0
            )
            route_bonus = (
                self.cross_route_bonus
                if doc_result is not None and graph_result is not None
                else 0.0
            )

            memory_content = doc_result.content if doc_result is not None else ""
            memory_metadata = (
                dict(doc_result.metadata)
                if doc_result is not None and isinstance(doc_result.metadata, dict)
                else {}
            )

            if not memory_content or not memory_metadata:
                memory = await self.memory_loader(doc_id)
                if not memory:
                    continue
                memory_content = str(memory.get("text") or memory_content)
                raw_metadata = memory.get("metadata") or memory_metadata
                memory_metadata = raw_metadata if isinstance(raw_metadata, dict) else {}

            final_score = min(
                1.0,
                self.document_route_weight * doc_signal
                + self.graph_route_weight * graph_signal
                + route_bonus,
            )

            score_breakdown: dict[str, float] = {}
            if doc_result and doc_result.score_breakdown:
                score_breakdown.update(doc_result.score_breakdown)
            if graph_result and graph_result.score_breakdown:
                score_breakdown.update(graph_result.score_breakdown)
            score_breakdown.update(
                {
                    "document_route_score": round(doc_signal, 4),
                    "graph_route_score": round(graph_signal, 4),
                    "cross_route_bonus": round(route_bonus, 4),
                    "dual_route_final_score": round(final_score, 4),
                }
            )
            if doc_result is not None:
                score_breakdown["document_keyword_score"] = round(
                    float(doc_result.bm25_score or 0.0),
                    4,
                )
                score_breakdown["document_vector_score"] = round(
                    float(doc_result.vector_score or 0.0),
                    4,
                )
            if graph_result is not None:
                score_breakdown["graph_keyword_score"] = round(
                    float(graph_result.keyword_score or 0.0),
                    4,
                )
                score_breakdown["graph_vector_score"] = round(
                    float(graph_result.vector_score or 0.0),
                    4,
                )

            merged_results.append(
                HybridResult(
                    doc_id=doc_id,
                    final_score=final_score,
                    rrf_score=max(
                        doc_result.rrf_score if doc_result is not None else 0.0,
                        graph_result.rrf_score if graph_result is not None else 0.0,
                    ),
                    bm25_score=doc_result.bm25_score
                    if doc_result is not None
                    else None,
                    vector_score=(
                        doc_result.vector_score if doc_result is not None else None
                    ),
                    content=memory_content,
                    metadata=memory_metadata,
                    score_breakdown=score_breakdown,
                )
            )

        merged_results.sort(key=lambda item: item.final_score, reverse=True)
        return merged_results[:k]


__all__ = ["DualRouteRetriever"]
