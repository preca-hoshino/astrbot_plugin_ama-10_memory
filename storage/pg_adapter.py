"""
asyncpg 兼容适配器
为存储层提供 aiosqlite 风格的 API 包装（? 占位符自动转 $N、cursor/row 接口）。
"""

from __future__ import annotations

import re
from typing import Any

import asyncpg

from astrbot.api import logger


def _convert_placeholders(sql: str) -> str:
    """将 SQLite 的 ? 占位符转换为 PostgreSQL 的 $1, $2, ...
    同时处理:
      - INSERT OR IGNORE → INSERT ... ON CONFLICT DO NOTHING
      - INSERT OR REPLACE → INSERT ... ON CONFLICT (...) DO UPDATE SET ...
    """
    was_insert_or_ignore = False

    # INSERT OR IGNORE → INSERT ... ON CONFLICT DO NOTHING
    if re.search(r'\bINSERT\s+OR\s+IGNORE\s+INTO\b', sql, flags=re.IGNORECASE):
        was_insert_or_ignore = True
        sql = re.sub(
            r'\bINSERT\s+OR\s+IGNORE\s+INTO\b',
            'INSERT INTO',
            sql,
            flags=re.IGNORECASE,
        )

    # INSERT OR REPLACE → INSERT INTO ... ON CONFLICT DO UPDATE
    # 注意: 调用方应显式写 ON CONFLICT DO UPDATE SET ... 以指定更新列
    # 这里只做基本的语法替换
    if re.search(r'\bINSERT\s+OR\s+REPLACE\s+INTO\b', sql, flags=re.IGNORECASE):
        sql = re.sub(
            r'\bINSERT\s+OR\s+REPLACE\s+INTO\b',
            'INSERT INTO',
            sql,
            flags=re.IGNORECASE,
        )
        # 如果没有显式 ON CONFLICT，追加基本的 DO UPDATE
        if 'ON CONFLICT' not in sql.upper():
            # 从 VALUES 子句前截断，追加 ON CONFLICT
            # 匹配 VALUES (...) 然后追加
            sql = re.sub(
                r'(VALUES\s*\([^)]*\))\s*$',
                r'\1 ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at',
                sql,
                flags=re.IGNORECASE | re.DOTALL,
            )

    # INSERT OR IGNORE 追加 ON CONFLICT DO NOTHING
    if was_insert_or_ignore and 'ON CONFLICT' not in sql.upper():
        if 'VALUES' in sql.upper():
            sql = sql.rstrip().rstrip(';') + ' ON CONFLICT DO NOTHING'

    counter = 0
    result = []
    in_string = False
    escape_next = False
    quote_char = None

    for ch in sql:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            result.append(ch)
            continue
        if in_string:
            result.append(ch)
            if ch == quote_char:
                in_string = False
            continue
        if ch in ("'", '"'):
            in_string = True
            quote_char = ch
            result.append(ch)
            continue
        if ch == "?":
            counter += 1
            result.append(f"${counter}")
        else:
            result.append(ch)

    return "".join(result)


class PgCursor:
    """模拟 aiosqlite.Cursor"""

    def __init__(self, conn: asyncpg.Connection):
        self._conn = conn
        self._rows: list[asyncpg.Record] | None = None
        self._status: str = ""
        self._lastrowid: int | None = None
        self._rowcount: int = 0

    @property
    def lastrowid(self) -> int | None:
        return self._lastrowid

    @property
    def rowcount(self) -> int:
        return self._rowcount

    @staticmethod
    def _coerce_params(params: tuple) -> tuple:
        """将 SQLite 风格参数转换为 asyncpg 兼容的原生类型

        注意: 不做 datetime 盲转——messages.timestamp 是 real 列，
        db_version.migrated_at 是 timestamptz 列，类型不同。
        datetime 转换应在调用方按列类型精确处理。
        """
        result = []
        for p in params:
            if isinstance(p, (dict, list)):
                import json
                result.append(json.dumps(p, ensure_ascii=False))
            else:
                result.append(p)
        return tuple(result)

    async def execute(self, sql: str, params: tuple = ()) -> PgCursor:
        pg_sql = _convert_placeholders(sql)
        params_list = list(self._coerce_params(params))

        # 检测是否是 RETURNING 查询 (INSERT ... RETURNING id)
        if "returning" in pg_sql.lower().split():
            row = await self._conn.fetchrow(pg_sql, *params_list)
            if row:
                self._rows = [row]
                self._lastrowid = row[0]
                self._rowcount = 1
            else:
                self._rows = []
                self._rowcount = 0
        elif pg_sql.strip().upper().startswith("SELECT"):
            self._rows = await self._conn.fetch(pg_sql, *params_list)
            self._rowcount = len(self._rows)
        else:
            status = await self._conn.execute(pg_sql, *params_list)
            self._status = status
            # 解析 INSERT 0 N / DELETE N / UPDATE N
            parts = status.split()
            if len(parts) >= 2:
                try:
                    self._rowcount = int(parts[-1])
                except ValueError:
                    self._rowcount = 0
            self._rows = None

        return self

    async def fetchone(self):
        if self._rows is not None and len(self._rows) > 0:
            row = self._rows[0]
            self._rows = self._rows[1:]
            return PgRow(row)
        return None

    async def fetchall(self):
        if self._rows is None:
            return []
        return [PgRow(r) for r in self._rows]

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    def __aiter__(self):
        self._iter_idx = 0
        return self

    async def __anext__(self):
        if self._rows is None or self._iter_idx >= len(self._rows):
            raise StopAsyncIteration
        row = PgRow(self._rows[self._iter_idx])
        self._iter_idx += 1
        return row


class PgRow:
    """模拟 aiosqlite.Row，支持 dict-style 和 index 访问"""

    def __init__(self, record: asyncpg.Record):
        self._record = record

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._record[key]
        return self._record[key]

    def __contains__(self, key):
        try:
            self._record[key]
            return True
        except (KeyError, IndexError):
            return False

    def keys(self):
        return self._record.keys()

    def __iter__(self):
        return iter(self._record)

    def __len__(self):
        return len(self._record)

    def __repr__(self):
        return repr(self._record)


class PgConnection:
    """模拟 aiosqlite.Connection"""

    def __init__(self, conn: asyncpg.Connection):
        self._conn = conn
        self.row_factory = None  # 不需要，asyncpg.Record 自带字段访问

    def execute(self, sql: str, params: tuple = ()):
        """返回一个对象，同时支持:
        - cursor = await conn.execute(...)        (awaitable)
        - async with conn.execute(...) as cursor:  (async context manager)
        """
        return _PgExecuteContext(self._conn, sql, params)

    async def commit(self):
        # asyncpg 默认 autocommit，无需操作
        # 如果在事务中，由上下文管理器处理
        pass

    async def close(self):
        # 不关闭底层连接（由连接池管理）
        pass

    def __getattr__(self, name):
        return getattr(self._conn, name)


class _PgExecuteContext:
    """让 PgConnection.execute() 的返回值同时支持 await 和 async with 两种用法。

    用法 1 (cursor = await conn.execute(sql, params)):
        __await__ → 执行 sql 并返回 PgCursor

    用法 2 (async with conn.execute(sql, params) as cursor):
        __aenter__ → 执行 sql 并返回 PgCursor
        __aexit__  → 无操作
    """

    def __init__(self, conn: asyncpg.Connection, sql: str, params: tuple):
        self._conn = conn
        self._sql = sql
        self._params = params
        self._cursor: PgCursor | None = None

    def __await__(self):
        # 支持 cursor = await conn.execute(sql, params) 模式
        return self._execute().__await__()

    async def _execute(self) -> PgCursor:
        cursor = PgCursor(self._conn)
        await cursor.execute(self._sql, self._params)
        self._cursor = cursor
        return cursor

    async def __aenter__(self) -> PgCursor:
        # 支持 async with conn.execute(sql, params) as cursor 模式
        if self._cursor is not None:
            return self._cursor  # 已经通过 await 执行过
        return await self._execute()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


class PgContextManager:
    """模拟 aiosqlite.connect() 的异步上下文管理器"""

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool
        self._conn: asyncpg.Connection | None = None
        self._pg_conn: PgConnection | None = None

    async def __aenter__(self) -> PgConnection:
        self._conn = await self._pool.acquire()
        self._pg_conn = PgConnection(self._conn)
        return self._pg_conn

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._conn:
            await self._pool.release(self._conn)
        return False


class PgPoolConnection:
    """连接池包装器: 每次操作从池中获取独立连接，避免 asyncpg 并发冲突。

    用法与 PgConnection/aiosqlite.Connection 完全兼容:
        conn = PgPoolConnection(pool)
        cursor = await conn.execute(sql, params)
        async with conn.execute(sql, params) as cursor:
            ...
        await conn.commit()   # no-op (autocommit)
        await conn.close()    # no-op (pool-managed)
    """

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool
        self.row_factory = None
        logger.debug("[PgPoolConnection] 创建连接池包装器")

    def execute(self, sql: str, params: tuple = ()):
        """返回 _PgPoolExecuteContext，每次调用从池获取独立连接"""
        return _PgPoolExecuteContext(self._pool, sql, params)

    async def commit(self):
        pass  # asyncpg autocommit

    async def close(self):
        pass  # 连接由池管理

    def __getattr__(self, name):
        raise AttributeError(
            f"PgPoolConnection 不支持属性 '{name}'。"
            f"如需直接操作 asyncpg.Connection，请使用 pool.acquire()。"
        )


class _PgPoolExecuteContext:
    """PgPoolConnection.execute() 的返回值，支持 await 和 async with 两种用法。

    每次操作从连接池获取独立连接，操作完成后归还，避免并发冲突。
    连接在 __aexit__ 或 __await__ 完成后释放。
    """

    def __init__(self, pool: asyncpg.Pool, sql: str, params: tuple):
        self._pool = pool
        self._sql = sql
        self._params = params
        self._cursor: PgCursor | None = None
        self._conn: asyncpg.Connection | None = None
        self._released: bool = False

    def __await__(self):
        return self._execute_and_release().__await__()

    async def _execute(self) -> PgCursor:
        self._conn = await self._pool.acquire()
        try:
            cursor = PgCursor(self._conn)
            await cursor.execute(self._sql, self._params)
            self._cursor = cursor
        except Exception:
            if not self._released:
                await self._pool.release(self._conn)
                self._released = True
            self._conn = None
            raise
        return self._cursor

    async def _execute_and_release(self) -> PgCursor:
        """await 模式：执行后立即释放连接，返回只读 PgCursor（数据已在内存中）。"""
        await self._execute()
        if self._conn is not None and not self._released:
            await self._pool.release(self._conn)
            self._released = True
            self._conn = None
        return self._cursor

    async def __aenter__(self) -> PgCursor:
        if self._cursor is not None:
            return self._cursor
        return await self._execute()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._conn is not None:
            await self._pool.release(self._conn)
            self._conn = None
        return False
