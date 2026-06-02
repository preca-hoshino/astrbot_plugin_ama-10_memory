"""
数据库迁移管理器 - PostgreSQL 模式存根
表结构由外部迁移脚本管理，此模块仅保留类签名以兼容 DecayScheduler 等引用。
"""

from __future__ import annotations

from typing import Any


class DBMigration:
    """数据库迁移管理器 (PG 模式下为 no-op)"""

    CURRENT_VERSION = 6

    VERSION_HISTORY = {
        1: "初始版本 - 基础记忆存储",
        2: "FTS5索引预处理",
        3: "会话ID迁移",
        4: "Schema v2 双通道总结字段",
        5: "Graph memory",
        6: "FTS 表前缀化",
    }

    def __init__(self, db_path: str):
        self.db_path = db_path

    async def get_db_version(self) -> int:
        return self.CURRENT_VERSION

    async def needs_migration(self) -> bool:
        return False

    async def migrate(self, progress_callback=None) -> dict[str, Any]:
        return {"success": True, "message": "PG 模式无需迁移", "duration": 0}

    async def create_backup(self) -> str | None:
        return None

    async def get_migration_info(self) -> dict[str, Any]:
        return {
            "current_version": self.CURRENT_VERSION,
            "latest_version": self.CURRENT_VERSION,
            "needs_migration": False,
        }
