"""
插件初始化器
负责插件的初始化逻辑
"""

import asyncio
import time
from pathlib import Path

from astrbot.api import logger
from astrbot.api.star import Context
from astrbot.core.provider.provider import EmbeddingProvider, Provider

from ..storage.conversation_store import ConversationStore
from .base.config_manager import ConfigManager
from .base.exceptions import InitializationError, ProviderNotReadyError
from .managers.conversation_manager import ConversationManager
from .managers.memory_engine import MemoryEngine
from .processors.memory_processor import MemoryProcessor
from .schedulers.decay_scheduler import DecayScheduler


class PluginInitializer:
    """插件初始化器"""

    def __init__(self, context: Context, config_manager: ConfigManager, data_dir: str):
        """
        初始化插件初始化器

        Args:
            context: AstrBot上下文
            config_manager: 配置管理器
            data_dir: 插件数据目录路径
        """
        self.context = context
        self.config_manager = config_manager
        self.data_dir = data_dir

        # 组件实例
        self.embedding_provider: EmbeddingProvider | None = None
        self.llm_provider: Provider | None = None
        self.vec_db = None
        self.graph_vec_db = None
        self.memory_engine: MemoryEngine | None = None
        self.memory_processor: MemoryProcessor | None = None
        self.conversation_manager: ConversationManager | None = None
        self.decay_scheduler: DecayScheduler | None = None

        # 初始化状态
        self._initialization_complete = False
        self._initialization_lock = asyncio.Lock()
        self._initialization_failed = False
        self._initialization_error: str | None = None
        self._providers_ready = False
        self._provider_check_attempts = 0
        self._max_provider_attempts = 60
        self._retry_task: asyncio.Task | None = None

    async def initialize(self) -> bool:
        """
        执行初始化

        Returns:
            bool: 是否初始化成功
        """
        async with self._initialization_lock:
            if self._initialization_complete or self._initialization_failed:
                return self._initialization_complete

        logger.info("AMA-10 Memory 插件开始后台初始化...")

        try:
            # 1. 等待 Provider 就绪
            if not await self._wait_for_providers_non_blocking():
                missing = []
                if not self.embedding_provider:
                    missing.append(
                        "Embedding Provider（请在 AstrBot 中配置向量嵌入模型）"
                    )
                if not self.llm_provider:
                    missing.append("LLM Provider（请在 AstrBot 中配置语言模型）")
                logger.warning(
                    f"以下 Provider 暂时不可用，将在后台继续尝试: {', '.join(missing)}"
                )
                self._start_retry_task_if_needed()
                return False

            # 2. Provider 就绪，继续完整初始化
            await self._complete_initialization()
            return True

        except Exception as e:
            logger.error(f"AMA-10 Memory 插件初始化失败: {e}", exc_info=True)
            self._initialization_failed = True
            self._initialization_error = str(e)
            return False

    def _start_retry_task_if_needed(self) -> None:
        """启动后台重试任务（避免重复启动）"""
        if self._retry_task and not self._retry_task.done():
            return

        self._retry_task = asyncio.create_task(self._retry_initialization())
        self._retry_task.add_done_callback(self._on_retry_task_done)

    def _on_retry_task_done(self, task: asyncio.Task) -> None:
        """重试任务完成回调，回收状态并记录异常"""
        self._retry_task = None
        if task.cancelled():
            return
        try:
            exc = task.exception()
            if exc:
                logger.error(f"Provider 重试任务异常退出: {exc}")
        except Exception:
            # 防御性处理：读取 task.exception() 时不应阻断主流程
            pass

    async def _wait_for_providers_non_blocking(self, max_wait: float = 5.0) -> bool:
        """非阻塞地检查 Provider 是否可用"""
        start_time = time.time()
        check_interval = 1.0

        while time.time() - start_time < max_wait:
            self._initialize_providers(silent=True)

            if self.embedding_provider and self.llm_provider:
                logger.info(
                    "Provider check passed: embedding and llm providers are ready."
                )
                self._providers_ready = True
                return True

            await asyncio.sleep(check_interval)
            self._provider_check_attempts += 1

        logger.debug(
            f"Provider 在 {max_wait}秒内未就绪（已尝试 {self._provider_check_attempts} 次）"
            f"：embedding={'ready' if self.embedding_provider else 'not ready'}, "
            f"llm={'ready' if self.llm_provider else 'not ready'}"
        )
        return False

    async def _retry_initialization(self):
        """后台重试初始化任务（指数退避策略）"""
        base_interval = 2.0
        max_interval = 30.0
        current_interval = base_interval
        log_interval = 5

        while (
            not self._initialization_complete
            and not self._initialization_failed
            and self._provider_check_attempts < self._max_provider_attempts
        ):
            await asyncio.sleep(current_interval)

            self._initialize_providers(silent=True)
            self._provider_check_attempts += 1

            if self._provider_check_attempts % log_interval == 0:
                missing = []
                if not self.embedding_provider:
                    missing.append("Embedding Provider")
                if not self.llm_provider:
                    missing.append("LLM Provider")
                logger.info(
                    f"等待 Provider 就绪（未就绪: {', '.join(missing)}）..."
                    f"（已尝试 {self._provider_check_attempts}/{self._max_provider_attempts} 次，"
                    f"下次重试间隔 {current_interval:.1f}s）"
                )

            if self.embedding_provider and self.llm_provider:
                logger.info(
                    f"Provider 在第 {self._provider_check_attempts} 次尝试后就绪，继续初始化。"
                )
                self._providers_ready = True

                try:
                    async with self._initialization_lock:
                        if not self._initialization_complete:
                            await self._complete_initialization()
                except Exception as e:
                    logger.error(f"重试初始化失败: {e}", exc_info=True)
                    self._initialization_failed = True
                    self._initialization_error = str(e)
                break

            # 指数退避，最大30秒
            current_interval = min(current_interval * 1.5, max_interval)

        if not self._initialization_complete and not self._initialization_failed:
            missing = []
            if not self.embedding_provider:
                missing.append("Embedding Provider（请配置向量嵌入模型）")
            if not self.llm_provider:
                missing.append("LLM Provider（请配置语言模型）")
            logger.error(
                f"以下 Provider 在 {self._provider_check_attempts} 次尝试后仍未就绪，初始化失败: "
                f"{', '.join(missing) if missing else '未知'}"
            )
            self._initialization_failed = True
            self._initialization_error = (
                "Provider 初始化超时。"
                f"未就绪 Provider: {', '.join(missing) if missing else '未知'}。"
                "请检查 provider_settings 配置和 AstrBot 默认 Provider。"
            )

    def _initialize_providers(self, silent: bool = False):
        """初始化 Embedding 和 LLM provider"""
        # 初始化 Embedding Provider
        emb_id = self.config_manager.get("provider_settings.embedding_provider_id")
        if emb_id:
            provider = self._get_provider_by_id(emb_id, silent=silent)
            if provider and isinstance(provider, EmbeddingProvider):
                self.embedding_provider = provider
                if not silent:
                    logger.info(f"成功从配置加载 Embedding Provider: {emb_id}")
            elif provider and not silent:
                logger.warning(f"Provider {emb_id} 不是 EmbeddingProvider 类型")

        if not self.embedding_provider:
            embedding_providers = self.context.get_all_embedding_providers()
            if embedding_providers:
                self.embedding_provider = embedding_providers[0]
                if not silent:
                    provider_id = getattr(
                        self.embedding_provider.provider_config,
                        "id",
                        self.embedding_provider.provider_config.get("id", "unknown"),
                    )
                    logger.info(f"未指定 Embedding Provider，使用默认的: {provider_id}")
            else:
                self.embedding_provider = None
                if not silent:
                    logger.debug("没有可用的 Embedding Provider")

        # 初始化 LLM Provider
        self.llm_provider = None
        llm_id = self.config_manager.get("provider_settings.llm_provider_id")
        if llm_id:
            provider = self._get_provider_by_id(llm_id, silent=silent)
            if provider and isinstance(provider, Provider):
                self.llm_provider = provider
                if not silent:
                    logger.info(f"成功从配置加载 LLM Provider: {llm_id}")
            elif provider and not silent:
                logger.warning(
                    f"Provider {llm_id} 不是聊天 Provider 类型，已忽略该配置。"
                )

        if not self.llm_provider:
            try:
                if silent and not self.context.get_all_providers():
                    self.llm_provider = None
                    return
                default_provider = self.context.get_using_provider()
                if default_provider and not isinstance(default_provider, Provider):
                    if not silent:
                        logger.warning(
                            "AstrBot 默认 Provider 类型不正确，期望聊天 Provider。"
                        )
                    self.llm_provider = None
                else:
                    self.llm_provider = default_provider
                if not silent and self.llm_provider:
                    logger.info("使用 AstrBot 当前默认的 LLM Provider。")
            except (ValueError, Exception) as e:
                if not silent:
                    logger.debug(f"获取默认 LLM Provider 失败: {e}")
                self.llm_provider = None

    def _get_provider_by_id(self, provider_id: str, *, silent: bool):
        """静默检查阶段绕过会打印 warning 的 AstrBot 查询接口。"""
        if not provider_id:
            return None
        if not silent:
            return self.context.get_provider_by_id(provider_id)
        provider_manager = getattr(self.context, "provider_manager", None)
        inst_map = getattr(provider_manager, "inst_map", None)
        if isinstance(inst_map, dict):
            return inst_map.get(provider_id)
        return None

    async def _complete_initialization(self):
        """完成完整的初始化流程"""
        if self._initialization_complete:
            return

        logger.info("开始完整初始化流程...")

        try:
            # 初始化数据库
            data_dir_path = Path(self.data_dir)
            graph_memory_enabled = self.config_manager.get("graph_memory.enabled", True)

            logger.debug(f"[Initializer] data_dir={self.data_dir}, graph_enabled={graph_memory_enabled}")
            logger.info(f"[Initializer] Embedding Provider: {getattr(self.embedding_provider, 'model', 'unknown')}")
            logger.info(f"[Initializer] LLM Provider: {getattr(self.llm_provider, 'model', 'unknown')}")

            if not self.embedding_provider:
                raise ProviderNotReadyError("Embedding Provider 未初始化")
            if not self.llm_provider or not isinstance(self.llm_provider, Provider):
                raise ProviderNotReadyError("LLM Provider 未初始化或类型不正确")

            # 检查是否使用 PostgreSQL
            pg_dsn = self.config_manager.get("database_settings.pg_dsn", "")
            if not pg_dsn:
                raise InitializationError("未配置 PostgreSQL 连接字符串 (database_settings.pg_dsn)")

            # PostgreSQL 模式
            from ..storage.pg_connection import init_pool
            from ..storage.pg_vec_db import PgVecDB

            await init_pool(pg_dsn)
            logger.info(f"PostgreSQL 连接池已初始化: {pg_dsn.split('@')[1] if '@' in pg_dsn else pg_dsn}")

            # provider_getter 回调: 让 PgVecDB 动态获取最新的 embedding_provider
            _get_emb = lambda: self.embedding_provider

            logger.info(f"[Initializer] 创建 PgVecDB (documents_vec, dim={self.embedding_provider.get_dim()})")
            self.vec_db = PgVecDB(
                vec_table="documents_vec",
                doc_table="documents",
                dimension=self.embedding_provider.get_dim(),
                embedding_provider=self.embedding_provider,
                provider_getter=_get_emb,
            )
            self.graph_vec_db = None
            if graph_memory_enabled:
                self.graph_vec_db = PgVecDB(
                    vec_table="graph_documents_vec",
                    doc_table="graph_documents",
                    dimension=self.embedding_provider.get_dim(),
                    embedding_provider=self.embedding_provider,
                    provider_getter=_get_emb,
                )

            logger.info(f"数据库已初始化。数据目录: {self.data_dir}")

            # 初始化MemoryEngine
            stopwords_dir = data_dir_path / "stopwords"
            stopwords_dir.mkdir(parents=True, exist_ok=True)

            memory_engine_config = {
                "rrf_k": self.config_manager.get("fusion_strategy.rrf_k", 60),
                "decay_rate": self.config_manager.get(
                    "importance_decay.decay_rate", 0.01
                ),
                "importance_weight": self.config_manager.get(
                    "recall_engine.importance_weight", 1.0
                ),
                "fallback_enabled": self.config_manager.get(
                    "recall_engine.fallback_to_vector", True
                ),
                "cleanup_days_threshold": self.config_manager.get(
                    "forgetting_agent.cleanup_days_threshold", 30
                ),
                "cleanup_importance_threshold": self.config_manager.get(
                    "forgetting_agent.cleanup_importance_threshold", 0.3
                ),
                "auto_cleanup_enabled": self.config_manager.get(
                    "forgetting_agent.auto_cleanup_enabled", True
                ),
                "stopwords_path": str(stopwords_dir),
                "graph_memory_enabled": graph_memory_enabled,
                "document_route_weight": self.config_manager.get(
                    "graph_memory.document_route_weight", 0.65
                ),
                "graph_route_weight": self.config_manager.get(
                    "graph_memory.graph_route_weight", 0.35
                ),
                "cross_route_bonus": self.config_manager.get(
                    "graph_memory.cross_route_bonus", 0.08
                ),
                "graph_expansion_limit": self.config_manager.get(
                    "graph_memory.expansion_limit", 24
                ),
                "graph_max_topics": self.config_manager.get(
                    "graph_memory.max_topics_per_memory", 6
                ),
                "graph_max_participants": self.config_manager.get(
                    "graph_memory.max_participants_per_memory", 8
                ),
                "graph_max_facts": self.config_manager.get(
                    "graph_memory.max_facts_per_memory", 8
                ),
                "index_rebuild_batch_size": self.config_manager.get(
                    "index_rebuild_settings.batch_size", 50
                ),
                "index_rebuild_embedding_batch_size": self.config_manager.get(
                    "index_rebuild_settings.embedding_batch_size", 8
                ),
                "index_rebuild_tasks_limit": self.config_manager.get(
                    "index_rebuild_settings.tasks_limit", 1
                ),
                "index_rebuild_max_retries": self.config_manager.get(
                    "index_rebuild_settings.max_retries", 5
                ),
                "index_rebuild_retry_base_delay": self.config_manager.get(
                    "index_rebuild_settings.retry_base_delay", 30.0
                ),
                "index_rebuild_batch_delay": self.config_manager.get(
                    "index_rebuild_settings.batch_delay", 5.0
                ),
                "index_rebuild_request_delay": self.config_manager.get(
                    "index_rebuild_settings.request_delay", 5.0
                ),
                "index_rebuild_max_failure_ratio": self.config_manager.get(
                    "index_rebuild_settings.max_failure_ratio", 0.02
                ),
            }

            self.memory_engine = MemoryEngine(
                vec_db=self.vec_db,
                graph_vector_db=self.graph_vec_db,
                llm_provider=self.llm_provider,
                config=memory_engine_config,
            )
            await self.memory_engine.initialize()
            logger.info("MemoryEngine 已初始化")

            # 初始化 ConversationManager
            logger.info("[Initializer] 初始化 ConversationManager...")
            conversation_db_path = data_dir_path / "conversations.db"
            conversation_store = ConversationStore(str(conversation_db_path))
            await conversation_store.initialize()

            session_config = self.config_manager.session_manager
            self.conversation_manager = ConversationManager(
                store=conversation_store,
                max_cache_size=session_config.get("max_sessions", 100),
                context_window_size=session_config.get("context_window_size", 50),
                session_ttl=session_config.get("session_ttl", 3600),
            )
            logger.info("ConversationManager 已初始化")

            # 自动修复 message_count 不一致问题
            await self._repair_message_counts(conversation_store)

            # 初始化 MemoryProcessor
            # 注意：MemoryProcessor 不直接持有 llm_provider 实例引用，
            # 而是在每次调用时通过 AstrBot 上下文动态解析 Provider，
            # 以避免 AstrBot 重新创建 Provider 后旧实例的 httpx client 被关闭
            # 导致的 "Cannot send a request, as the client has been closed" 错误。
            llm_id = self.config_manager.get("provider_settings.llm_provider_id")
            self.memory_processor = MemoryProcessor(
                self.context, llm_provider=llm_id if llm_id else None
            )
            logger.info("MemoryProcessor 已初始化")

            # 异步初始化 TextProcessor
            if self.memory_engine and hasattr(self.memory_engine, "text_processor"):
                if self.memory_engine.text_processor and hasattr(
                    self.memory_engine.text_processor, "async_init"
                ):
                    await self.memory_engine.text_processor.async_init()
                    logger.info("TextProcessor 停用词已加载")

            # 启动重要性衰减调度器
            decay_rate = self.config_manager.get("importance_decay.decay_rate", 0.01)
            auto_cleanup_enabled = self.config_manager.get(
                "forgetting_agent.auto_cleanup_enabled", True
            )
            if self.memory_engine and (decay_rate > 0 or auto_cleanup_enabled):
                backup_enabled = self.config_manager.get(
                    "backup_settings.enabled", True
                )
                backup_keep_days = self.config_manager.get(
                    "backup_settings.keep_days", 7
                )
                scheduler = DecayScheduler(
                    memory_engine=self.memory_engine,
                    decay_rate=decay_rate,
                    data_dir=self.data_dir,
                    db_migration=None,
                    backup_enabled=backup_enabled,
                    backup_keep_days=backup_keep_days,
                )
                await scheduler.start()
                self.decay_scheduler = scheduler
                logger.info("DecayScheduler 已启动")

            # 标记初始化完成
            self._initialization_complete = True
            logger.info("AMA-10 Memory 插件初始化成功。")
            logger.info(f"[Initializer] 组件状态: memory_engine={self.memory_engine is not None}, "
                        f"memory_processor={self.memory_processor is not None}, "
                        f"conversation_manager={self.conversation_manager is not None}, "
                        f"graph_vec_db={self.graph_vec_db is not None}, "
                        f"decay_scheduler={self.decay_scheduler is not None}")

        except Exception as e:
            logger.error(f"完整初始化流程失败: {e}", exc_info=True)
            self._initialization_failed = True
            self._initialization_error = str(e)
            raise InitializationError(f"初始化失败: {e}") from e

    async def _repair_message_counts(self, conversation_store: ConversationStore):
        """修复会话表中 message_count 与实际消息数量不一致的问题"""
        try:
            logger.info("开始检查并修复 message_count 一致性。")
            fixed_sessions = await conversation_store.sync_message_counts()

            if fixed_sessions:
                logger.info(f"已修复 {len(fixed_sessions)} 个会话的 message_count")
            else:
                logger.debug("所有会话的 message_count 均正确")

        except Exception as e:
            logger.error(f"修复 message_count 失败: {e}", exc_info=True)

    @property
    def is_initialized(self) -> bool:
        """是否已初始化"""
        return self._initialization_complete

    @property
    def is_failed(self) -> bool:
        """是否初始化失败"""
        return self._initialization_failed

    @property
    def error_message(self) -> str | None:
        """错误消息"""
        return self._initialization_error

    async def ensure_initialized(self, timeout: float = 30.0) -> bool:
        """
        确保插件已初始化

        Args:
            timeout: 超时时间（秒）

        Returns:
            bool: 是否初始化成功
        """
        if self._initialization_complete:
            return True

        if self._initialization_failed:
            return False

        # 等待初始化完成
        start_time = time.time()
        while not self._initialization_complete and not self._initialization_failed:
            if time.time() - start_time > timeout:
                logger.error(f"等待插件初始化超时（{timeout}秒）")
                return False
            await asyncio.sleep(0.2)

        return self._initialization_complete

    async def stop_scheduler(self) -> None:
        """停止衰减调度器"""
        if self.decay_scheduler:
            await self.decay_scheduler.stop()
            self.decay_scheduler = None

    async def stop_background_tasks(self) -> None:
        """停止初始化阶段的后台任务（如Provider重试）"""
        if self._retry_task and not self._retry_task.done():
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass
        self._retry_task = None

    async def close_pg_pool(self) -> None:
        """关闭 PostgreSQL 连接池"""
        from ..storage.pg_connection import close_pool
        await close_pool()
