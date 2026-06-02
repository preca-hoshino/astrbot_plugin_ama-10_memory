"""
PostgreSQL 连接池管理
提供全局 asyncpg 连接池，供所有存储层共享使用。
"""

from __future__ import annotations

import asyncpg

from astrbot.api import logger

_pool: asyncpg.Pool | None = None

PG_SCHEMA = "livingmemory"


async def init_pool(dsn: str, *, min_size: int = 2, max_size: int = 10) -> asyncpg.Pool:
    """初始化全局连接池"""
    global _pool
    if _pool is not None:
        return _pool

    _pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=min_size,
        max_size=max_size,
        command_timeout=60,
        server_settings={"search_path": f"{PG_SCHEMA},public"},
    )
    logger.info(f"[PG] 连接池已创建 (min={min_size}, max={max_size}, schema={PG_SCHEMA})")
    # 验证连接可用性
    async with _pool.acquire() as conn:
        pg_version = await conn.fetchval("SELECT version()")
        logger.info(f"[PG] 连接验证成功: {pg_version[:60]}...")
        table_count = await conn.fetchval(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = $1", PG_SCHEMA
        )
        logger.info(f"[PG] schema '{PG_SCHEMA}' 中有 {table_count} 张表")
    return _pool


def get_pool() -> asyncpg.Pool:
    """获取全局连接池"""
    if _pool is None:
        raise RuntimeError("PostgreSQL 连接池未初始化，请先调用 init_pool()")
    return _pool


async def close_pool() -> None:
    """关闭全局连接池"""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("[PG] 连接池已关闭")


def is_pg_mode() -> bool:
    """是否已初始化 PG 连接池"""
    return _pool is not None
