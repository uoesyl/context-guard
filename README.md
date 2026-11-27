# Context Guard

**为 [Antigravity](https://antigravity.google) 准备的上下文守护插件：实时上下文用量仪表盘、按模型自校准的压缩前预警，以及压缩后“不失忆”的无损恢复。**

> English summary: Antigravity compacts long conversations into a lossy summary. Context Guard watches the
> real prompt size, warns the agent before compaction so it saves its working memory (`task_state.md`), and
> right after a compaction injects a recovery pack with the user's verbatim requests, the current plan,
> `task_state.md` and edited files. It ships a side-pane meter, a `recall.py` CLI that queries the lossless
> `transcript_full.jsonl`, and context-usage statistics. Messages are in Chinese.

## 功能

| 模块 | 作用 |
|---|---|
| **压缩前预警**（`hooks/guard.py`，PreInvocation） | 按真实 prompt token 计算用量；在压缩目标的约 68% / 82% 时向代理注入一级预警 / 二级紧急，要求先更新 `task_state.md`。阈值按模型自校准（实测自动压缩点 > 内置值 > 会话数据库上限 > 默认 512k）。 |
| **压缩后恢复包**（PreInvocation + PreToolUse） | 检测到 `CHECKPOINT`（自动压缩或 `/compact`）后注入：用户原话（逐字）、最近的执行计划、多代理消息、`task_state.md`、本会话改过的文件和召回命令。压缩后第一次修改类工具调用会被拦截一次，确保代理先读恢复包；并校验恢复包确实写进了记录，丢失时自动重发。 |
| **工作记忆提醒** | 恢复包送达后 8 次模型调用内 `task_state.md` 仍未更新，提醒一次。 |
| **Context 侧边栏**（`sidecars/context_meter`） | 实时显示上下文用量、预警刻度、按模型的缓存命中 / token / TPS 统计、压缩历史、上下文组成与工具 / Skill / MCP 使用；随窗口宽度自适应（窄栏单列，≥900px 双列，≥1500px 三列）。 |
| **召回 CLI**（`scripts/recall.py`） | 查询完整无损的 `transcript_full.jsonl`：`users` / `grep` / `step` / `range` / `files` / `messages`。 |
| **用量分类**（`scripts/context_stats.py`） | 按来源（基线、工具结果、工具参数、思考、通知、Hook 注入等）估算当前窗口和全对话的 token 占用。 |
| **规则与技能** | `rules/AGENTS.md`（维护 `task_state.md` 的工作规范）和 `skills/context-recovery`（压缩后如何核实旧细节）。 |

## 环境要求

- Antigravity（支持 plugins、hooks、sidecars 的版本）
- Python 3.9+（hook 与 CLI 仅用标准库）
- Node.js 22.13+（侧边栏使用内置 `node:sqlite`；推荐 Node 24）
- Windows / macOS / Linux 均可；开发与实测主要在 Windows

## 安装

1. 克隆到 Antigravity 的插件目录（目录名需为 `context-guard`）：

   ```bash
   # Windows PowerShell
   git clone https://github.com/uoesyl/context-guard.git "$HOME/.gemini/config/plugins/context-guard"
   # macOS / Linux
   git clone https://github.com/uoesyl/context-guard.git ~/.gemini/config/plugins/context-guard
   ```

2. 注册全局 hook（写入 `~/.gemini/config/hooks.json`，保留你已有的其他 hook，自动备份为 `hooks.json.bak`）：

   ```bash
   python ~/.gemini/config/plugins/context-guard/install.py
   # 先预览：加 --dry-run；指定解释器：--python /path/to/python
   ```

   > 为什么不用插件自带的 `hooks.json`：插件内的 hook 不会在 swarm worker 中运行，而恢复包需要覆盖所有会话。
   > `hooks.json` 必须是**无 BOM 的 UTF-8**，否则 Antigravity 会静默忽略整个文件；`install.py` 会处理这一点。

3. 在 Antigravity 的 Customizations 页面确认 **Context Guard** 已启用，然后在侧边栏打开 **Context** 面板。新会话立即生效，进行中的会话在下一次模型调用时生效。

### 手动注册 hook

不想运行 `install.py` 时，可在 `~/.gemini/config/hooks.json` 中加入（把路径换成你的实际路径）：

```json
{
  "context-guard": {
    "PreInvocation": [
      { "type": "command", "command": "python /home/you/.gemini/config/plugins/context-guard/hooks/guard.py", "timeout": 10 }
    ],
    "PreToolUse": [
      { "matcher": "*", "hooks": [
        { "type": "command", "command": "python /home/you/.gemini/config/plugins/context-guard/hooks/guard.py", "timeout": 10 }
      ] }
    ]
  }
}
```

## 使用

插件安装后自动工作，代理会按 `rules/AGENTS.md` 维护 `task_state.md`。常用手动操作：

```bash
R=~/.gemini/config/plugins/context-guard/scripts/recall.py
python $R --conv <会话ID> users                 # 全部用户原话
python $R --conv <会话ID> grep "学习率|batch"    # 正则搜索
python $R --conv <会话ID> range 100 180 --brief # 某段步骤概览
python $R --conv <会话ID> step 137              # 某一步全文
python ~/.gemini/config/plugins/context-guard/scripts/context_stats.py [<会话ID> | --conv <会话ID> | --transcript <路径>] [--json]  # 无参默认分析最新会话
```

侧边栏的「复制 /compact」和「提醒更新记忆」按钮可以在自然断点手动压缩，或让代理立即更新 `task_state.md`。

### 阈值配置

阈值保存在 `~/.gemini/antigravity/plugin_data/context-guard/calibration.json`，会随实际自动压缩自动校准。需要手动指定某个模型的压缩目标时，在对应模型条目下加 `"override": <token数>`：

```json
{ "models": { "claude-opus-4": { "override": 200000 } } }
```

重建校准数据：`python hooks/guard.py --calibrate`。数据目录可用环境变量 `CONTEXT_GUARD_DATA` 改到别处。

## 数据与隐私

- 全部在本机运行，不联网、不上传任何内容。
- 只读访问 Antigravity 的会话记录（`~/.gemini/antigravity/brain/<id>/.system_generated/logs/transcript_full.jsonl`）和会话数据库（只读模式）。
- 插件自身只写 `~/.gemini/antigravity/plugin_data/context-guard/`（状态、日志 `guard.log`、校准、计划副本）。
- hook 内部任何异常都不会阻塞代理：PreInvocation 返回 `{}`，PreToolUse 返回 `ask`（遵循你的权限设置），错误写入 `guard.log`。

## 已知限制

- **自动压缩与手动 `/compact` 的注入时机差异**：自动压缩发生在 `PreInvocation` 之后、模型调用之前，因此压缩后的首轮模型在只有有损摘要的情况下运行，由 `PreToolUse` 在首个修改类工具调用时拦截并注入完整恢复包（若该轮没有调用修改类工具，则在下一轮 `PreInvocation` 注入）；手动 `/compact` 在下一轮用户消息前已写入 `CHECKPOINT`，因此下次 `PreInvocation` 即可直接注入恢复包。
- 仪表盘的按步耗时 / TPS 依赖 Antigravity 会话数据库的内部格式，版本升级后可能需要适配。
- 编辑 / 重试旧消息（回退）后，被丢弃分支上的模型调用不计入统计，面板会单独显示其数量。
- 提示文字目前为中文。

## 测试

```bash
python tests/test_guard.py                       # hook 与 CLI 单元测试（加 --replay-local 回放本机历史会话）
cd sidecars/context_meter && node --test meter.test.mjs
```

## 卸载

```bash
python ~/.gemini/config/plugins/context-guard/install.py --uninstall
```

然后在 Customizations 页面卸载插件，或直接删除插件目录；数据目录 `~/.gemini/antigravity/plugin_data/context-guard/` 可一并删除。

## 许可

[MIT](LICENSE)
