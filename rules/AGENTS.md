# Context Guard：抵御自动压缩的工作记忆

Antigravity 会在上下文接近当前模型的窗口时自动压缩（Gemini 约 512k；其他模型窗口不同，Context Guard 按模型自校准阈值预警），摘要有损。按以下规则维护工作记忆：

1. **task_state.md**：多步任务（预计超过约 10 次工具调用，或跨多轮对话）开始时，在当前会话的 artifact 目录下创建 `task_state.md`（非用户可见的 artifact，不要求用户审阅），固定章节：
   - 目标
   - 用户原话约束（逐字引用，不改写）
   - 已做决定及原因
   - 已完成
   - 待办
   - 关键路径、命令与数值
2. **何时更新**：完成一个里程碑、用户提出新要求或修正、收到 "Context Guard 预警" 系统消息时。保持精简（≤ 6KB），只记录压缩后必须知道的东西。
3. **压缩之后**：看到 "Resuming from a compaction" 或 "Context Guard：第 N 步发生了自动压缩" 时，先读恢复包和 `task_state.md`，再继续。不在两者之中、又没有把握的旧细节，用本插件 `context-recovery` 技能里的 recall.py 查原文核实，不要猜。
4. 收到预警后，如果用户在线，在下次回复用户时提示：可以在自然断点手动执行 `/compact`。
5. **节省上下文与精准读码（Windows 上默认 Shell 通常是 Windows PowerShell 5.1；已安装 `rg`/`fd` 时优先使用）**：读文件是上下文的主要消耗。
   - **先定位后读码**：定位文件或符号优先用 `fd --max-results 50`、`rg -n -m 20 --max-columns 160`，或用 `rg -n "^(class |def |async def |function |export )" <file>` 先看大纲；**未通过搜索定位的探索性 `read_file`，单轮并行不超过 3 个文件**。仅在**跨多模块全仓审计或陌生大范围摸底**时派 `research` 子代理隔离上下文；常规开发、局部排障与已知范围修改留在主会话，避免子代理无法复用主会话前缀缓存反而推高总消耗。
   - **防切片通读且防漏定义**：>250 行文件优先按行号区间读目标完整语义块（单次 ≤ 250–300 行），按需补读文件头 `1–50` 行（imports/类型/常量）或所属类头避免漏定义；**禁止对同一大文件连续无目的顺读超过 2 段**，不要反复整文件重读；并在 `task_state.md` 记录关键代码位置（`文件:行号 + 用途`），压缩后据此定点读取。
   - **Plan → Execute 零重复预读**：规划阶段在 `exit_plan_mode` 计划中写明精确文件路径、函数/符号名与行号区间；执行阶段直接按计划区间定点 `read_file` 修改，禁止对已读清的文件再大面积重读。
   - **命令按场景限流（Windows 下严禁 `head`/`tail`/`grep`）**：Windows 上 `run_command` 后端通常为 Windows PowerShell 5.1，无 `head`/`tail`/`grep`（需用 PowerShell 7 时显式调 `pwsh`；Linux/macOS 用对应原生命令）。检索/列表在源头限流（`rg -m`、`fd --max-results`、`git log -n 20 --oneline`、`git diff --stat`）；单测/报错日志**不切头部**，用 `| Select-Object -Last 60` 或精简模式抓尾部失败点，脚本诊断只打印计数与摘要，不裸吐大 JSON。
6. 如果修改类工具被 "【Context Guard 拦截】" 拒绝：说明刚发生压缩（自动或 /compact），那次操作没有执行。先读拦截信息里的恢复包，核对后再重新发起。
7. 写配置文件（尤其 `~/.gemini/config/*.json`）、`scratch/` 日志与 JSON 产物时，一律写成**无 BOM 的严格 UTF-8**：优先用 `write_file`、Python `Path.write_text(..., encoding="utf-8")`、`pwsh`（PowerShell 7），或 `[IO.File]::WriteAllText(路径, 内容, [Text.UTF8Encoding]::new($false))`。严禁在 Windows PowerShell 5.1 中直接用 `>` / `>>` / `Tee-Object`（默认写出 UTF-16LE）或 `Set-Content -Encoding utf8` / `Out-File -Encoding utf8`（会加 UTF-8 BOM，导致 Antigravity 静默忽略 `hooks.json` 或严格 UTF-8 校验报错）。
8. **多会话并发归属与诚实记账**：发现 `HEAD`、工作区脏文件或数据摘要出现非本会话改动时，先只读核查并行会话与后台进程归属，严禁盲目回滚或跨会话抢改共享文件；若工作区已由新会话接管，立即切入**只读停驻**，在 `task_state.md` 首部明写 `状态：已停驻（时刻），等 <owner会话/任务> 完成 <具体事项>` 并列出移交清单与跨会话用户原话。所有新增 `scratch` 产物须在 `task_state.md` 注明用途、结论归属与核查边界；未通过或未复测的硬指标即使获用户个案豁免，也只能记为“未通过/未验证 + 个案人工豁免”，严禁写成验收通过。
9. **上下文用量分类**：用户问"什么占用了上下文 / 工具和 skill 用了多少"时，运行 `python "$HOME/.gemini/config/plugins/context-guard/scripts/context_stats.py" <会话ID>`（插件装在别处时换成实际路径）（`--json` 输出机器可读），按来源（基线、各工具结果、各工具调用参数、模型思考、通知、Hook 注入等）给出当前窗口与全对话的 token 估算和精确字符数，以及工具、Skill、MCP 使用统计；预警消息里也会附上本窗口前几大来源。
