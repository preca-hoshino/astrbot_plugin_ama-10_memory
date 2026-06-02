# AMA-10 Memory

> 为 [AstrBot](https://github.com/Soulter/AstrBot) 打造的智能长期记忆插件，拥有完整的记忆生命周期管理、图记忆、多路召回等能力。

**本项目 Fork 自 [lxfight/astrbot_plugin_livingmemory](https://github.com/lxfight-s-Astrbot-Plugins/astrbot_plugin_livingmemory)，基于 AGPL-3.0 协议发布。感谢原作者 [lxfight](https://github.com/lxfight) 的贡献。**

---

## ✨ 特性

- **完整的记忆生命周期** — 从对话中自动提取、评估、存储、衰减、遗忘
- **图记忆** — 基于知识图谱的结构化记忆，支持实体关系抽取与推理
- **多路召回** — BM25 + 向量检索 + 图检索，RRF 融合排序
- **LLM Tool 集成** — 记忆写入 (`memorize`) 和检索 (`memory_search`) 作为 LLM 工具，让 Agent 主动管理记忆
- **LLM 上下文注入** — 在 `on_llm_request` 时自动注入相关记忆到 system prompt
- **会话感知** — 区分群聊/私聊，按会话隔离记忆
- **WebUI 管理** — 通过 AstrBot 仪表盘可视化管理记忆、图谱、召回调试
- **PostgreSQL 支持** — 可选 pgvector 后端，适合大规模部署
- **多语言** — 支持中文、英文、俄文

## 📦 安装

### 前置要求

- AstrBot >= 4.24.2
- PostgreSQL >= 14 with pgvector 扩展

### 方式一：AstrBot 插件市场

在 AstrBot 插件页面搜索 `AMA-10 Memory` 安装。

### 方式二：手动安装

```bash
cd /path/to/AstrBot/data/plugins
git clone https://github.com/preca-hoshino/astrbot_plugin_ama-10_memory.git astrbot_plugin_ama_10_memory
```

然后在 AstrBot 管理面板中启用插件，重启 AstrBot。

### 依赖

```bash
pip install networkx jieba pytz aiofiles asyncpg
```

### PostgreSQL 配置

本插件**仅支持 PostgreSQL** (需要 pgvector 扩展)。安装 pgvector：

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## 🔧 配置

在 AstrBot 插件管理面板中配置，或直接编辑配置文件：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `database_settings.pg_dsn` | PostgreSQL 连接 URL | `""` (必填) |
| `provider_settings.llm_provider_id` | 记忆摘要用的 LLM 模型 ID | (AstrBot 默认) |
| `provider_settings.embedding_provider_id` | 向量化用的 Embedding 模型 ID | (AstrBot 默认) |
| `session_manager.*` | 会话管理配置 | — |
| `memory_engine.*` | 记忆引擎配置 | — |
| `graph_memory.*` | 图记忆配置 | — |
| `retrieval.*` | 召回策略配置 | — |

详细配置说明请参考 `_conf_schema.json` 中的注释。

## 📖 使用

### 指令

| 指令 | 说明 |
|------|------|
| `/lmem status` | 查看系统状态 |
| `/lmem search <关键词> [数量]` | 搜索记忆（默认 5 条） |
| `/lmem forget <ID>` | 删除指定记忆 |
| `/lmem rebuild-index` | 重建索引（修复索引不一致） |
| `/lmem rebuild-graph` | 重建图记忆索引 |
| `/lmem webui` | 查看 WebUI 管理界面入口 |
| `/lmem summarize` | 立即触发当前会话的记忆总结 |
| `/lmem reset` | 重置当前会话记忆上下文 |
| `/lmem cleanup [preview\|exec]` | 清理历史消息中的记忆片段 |
| `/lmem help` | 显示帮助 |

### LLM Tools

插件向 LLM 注册了两个工具：

- **`memorize`** — Agent 主动写入记忆（知识、偏好、事件等）
- **`memory_search`** — Agent 主动检索相关记忆

### 自动上下文注入

插件在每次 LLM 请求前，自动检索与当前对话相关的记忆并注入到 system prompt 中，无需手动操作。

## 🏗️ 架构

```
main.py                     # 插件入口
storage/                    # PostgreSQL 存储层
├── pg_connection.py        # PG 连接池管理
├── pg_adapter.py           # asyncpg 兼容适配器
├── pg_vec_db.py            # pgvector 向量数据库
├── graph_store.py          # 图记忆存储
├── atom_store.py           # Atom 存储
├── conversation_store.py   # 会话存储
└── db_migration.py         # 数据库迁移
core/
├── base/                   # 基础配置、常量、异常
├── managers/
│   ├── memory_engine.py    # 记忆引擎（存储/检索/衰减）
│   ├── graph_memory_manager.py  # 图记忆管理
│   ├── conversation_manager.py  # 会话管理
│   ├── atom_lifecycle_manager.py # 记忆原子生命周期
│   └── backup_manager.py   # 数据备份
├── processors/
│   ├── memory_processor.py # 记忆处理（提取/评估）
│   ├── graph_extractor.py  # 图谱实体关系抽取
│   ├── entity_resolver.py  # 实体消歧
│   ├── atom_classifier.py  # 记忆分类
│   └── text_processor.py   # 文本预处理
├── retrieval/
│   ├── bm25_retriever.py   # BM25 检索
│   ├── vector_retriever.py # 向量检索
│   ├── graph_retriever.py  # 图检索
│   ├── hybrid_retriever.py # 混合检索
│   └── rrf_fusion.py       # RRF 融合排序
├── tools/
│   ├── memory_memorize_tool.py  # LLM Tool: memorize
│   └── memory_search_tool.py    # LLM Tool: memory_search
├── schedulers/
│   └── decay_scheduler.py  # 记忆衰减调度
├── i18n/                   # 多语言
├── prompts/                # LLM 提示词模板
├── event_handler.py        # 事件处理
├── command_handler.py      # 指令处理
└── plugin_initializer.py   # 插件初始化
```

## 📋 从 LivingMemory 迁移

如果你正在使用 `lxfight/astrbot_plugin_livingmemory`，迁移到 AMA-10 Memory 需要注意：

1. **数据库不兼容** — AMA-10 Memory 使用 `ama_10_memory.db` 等新文件名，旧数据不会自动迁移
2. **手动迁移** — 如需保留旧数据，可将 `livingmemory.db` 重命名为 `ama_10_memory.db`（表结构兼容）
3. **配置** — 配置格式基本兼容，但插件名称已更改

## 🙏 致谢

- [lxfight](https://github.com/lxfight) — 原始插件 [astrbot_plugin_livingmemory](https://github.com/lxfight-s-Astrbot-Plugins/astrbot_plugin_livingmemory) 的作者
- [AstrBot](https://github.com/Soulter/AstrBot) — 机器人框架
- [jieba](https://github.com/fxsjy/jieba) — 中文分词
- [NetworkX](https://networkx.org/) — 图计算

## 📄 许可证

本项目基于 [GNU Affero General Public License v3.0](LICENSE) 开源。

原始代码来自 [lxfight/astrbot_plugin_livingmemory](https://github.com/lxfight-s-Astrbot-Plugins/astrbot_plugin_livingmemory)（AGPL-3.0），本项目在同一协议下发布。
