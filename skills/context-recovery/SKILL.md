---
name: context-recovery
description: 压缩后恢复上下文、核实早先细节。看到 "Resuming from a compaction" 或 "Context Guard" 系统消息时、需要引用早先的数值/路径/用户要求但没把握时、用户说"我之前说过/你忘了"时使用。用 recall.py 查询完整无损的 transcript_full.jsonl。
---

# 压缩后的上下文恢复

Antigravity 会在上下文接近模型窗口时自动压缩（按模型自校准阈值，优先级为实测自动压缩点 > 内置阈值 > 会话数据库上限 > 默认 512k；Gemini 约 480k–524k token），把几十万字符压成约 1–3 万字符的摘要，细节大量丢失。完整原文仍保存在
`~/.gemini/antigravity/brain/<conversation-id>/.system_generated/logs/transcript_full.jsonl`。

## 流程

1. **先读恢复包**：context-guard 会在压缩后注入“用户原话（逐字）+ 当前执行计划（若有）+ 多代理消息（若有）+ `task_state.md` + 修改过的文件”。摘要与原话冲突时以原话为准。
2. **更新 task_state.md**（artifact 目录下）：把恢复包和摘要中的目标、约束、决定、待办合并进去。
3. **先查后答**：任何没把握的旧细节都先查原文，不要凭印象补全。
4. 查不到时明确告诉用户"记录中没有找到"，不要编造。

## recall.py

位于本插件的 `scripts/recall.py`（默认安装路径 `~/.gemini/config/plugins/context-guard/scripts/recall.py`；恢复包里会给出本机的完整命令）。用 `--conv <conversation-id>` 指定当前会话（或用 `--transcript <路径>`）。默认只读取当前活跃分支（自动过滤编辑/重试消息后被丢弃的旧分支）；如需排查被丢弃分支的内容，可加 `--all-branches`。

```powershell
$R = "$HOME/.gemini/config/plugins/context-guard/scripts/recall.py"   # bash: R=~/.gemini/config/plugins/context-guard/scripts/recall.py
python $R --conv <id> users                         # 全部用户原话
python $R --conv <id> grep "batch_size|学习率"       # 正则搜索（content/thinking/tool_calls）
python $R --conv <id> grep "config.toml" --type USER_INPUT --max 50
python $R --conv <id> range 100 180 --brief         # 某一段步骤的概览
python $R --conv <id> step 137                      # 某一步完整内容
python $R --conv <id> files                         # 写过/改过的文件
python $R --conv <id> messages [--sender 61f5]      # 代理间消息：收到的任务/回报、派发的 subagent、发出的消息、读到的 swarm 消息板
python $R --conv <id> --all-branches users          # 包含编辑/重试后被丢弃的旧分支
```

## 多代理场景

- **subagent / swarm worker**：没有用户输入，任务来自编排者的第一条消息。恢复包里的"编排者下发的任务说明"一节就是你的任务原文。
- **编排者**：恢复包会列出派发过的 subagent / worker 和它们的会话 ID；需要细节时用 `--conv <它们的会话ID>` 直接查它们的完整记录。
- **swarm**：消息板本身不会丢，完整历史用 `read_board{since:0}` 查看。

输出超过 2 万字符时会写入 `<conversation>/scratch/recall_*.txt`，再用 read_file 分段读取。

## 注意

- 压缩前的工具输出（文件内容、命令结果）可能已经过时，涉及当前状态时重新读取文件或重新运行命令，不要只看历史记录。
- 不要把 recall 结果整段贴给用户，只引用相关部分。
