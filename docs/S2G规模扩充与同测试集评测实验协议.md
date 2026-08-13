# S2G 规模扩充与同测试集评测实验协议

## Material Passport

- **Artifact ID**：`searchr1-s2g-scaleup-aligned-eval-v1`
- **Origin mode**：academic-research-suite / experiment-agent
- **Version**：`v1`
- **Frozen date**：2026-07-27
- **Status**：`ACTIVE / GATE S1 PASS / GATE S2 READY`
- **目标**：在 Search-R1 状态分布上把结构化 Judge 监督扩充到与
  S2G-RAG 原论文相近的规模，并在与作者公开代码一致的
  HotpotQA distractor validation 前 1,000 题上比较 Native Search-R1、
  Structured Base 和扩充版 S2G LoRA。
- **现有证据边界**：Core200 及其标签已经查看，只能作为开发回归集；
  不得用于本协议的 prompt、checkpoint、阈值或训练规模选择。
- **外部材料**：
  - S2G-RAG 论文：ACL 2026，Anthology ID `2026.acl-long.1185`；
  - 官方代码：`https://github.com/nianaaa/S2G-RAG`；
  - 本协议冻结的官方代码 commit：
    `5d842a67a0a99a7b545bbad0dc402ceaae0e5eff`。

## 1. 研究问题

本实验回答两个分层问题：

1. **状态判别问题**：扩充监督后的 Qwen3.5-2B S2G Judge 是否比同
   prompt 的 Structured Base 更准确地识别“现在可以安全提前停止”的
   Search-R1 状态？
2. **系统问题**：把该 Judge 接入 Search-R1 后，是否能在不降低官方
   EM 的前提下减少搜索次数？

同一模型在某个状态上是否能够被强制生成正确答案，与原生 Search-R1
最终答案生成路径是否完全一致，是独立的生成策略问题。本协议暂不以此
作为阻塞项，但通过分层指标避免把“学会停止”和“端到端系统变好”混为
一谈。

## 2. 与 S2G-RAG 原论文的对齐边界

### 2.1 可对齐部分

- 数据集：`hotpotqa/hotpot_qa`
- 配置：`distractor`
- 最终评测来源：官方 `validation`
- 公开代码评测子集：按原始顺序最先出现的 1,000 个唯一问题 ID
- 最大轮数：4
- Judge 输出：`sufficient + gap_items`
- LoRA：rank 16、alpha 32、dropout 0.05、3 epochs、学习率 `1e-4`
- 训练规模参照：2,804 个清洗后 turn-level snapshots

### 2.2 不对齐部分

| 组件 | S2G-RAG 原论文 | 本实验 |
|---|---|---|
| Reasoner | Llama-3-8B-Instruct | Search-R1 Qwen2.5-7B PPO |
| Judge backbone | Llama-3.2-3B-Instruct | Qwen3.5-2B |
| Teacher | GPT-4o-mini | 本地 Qwen3.6-35B-A3B |
| 检索宽度 | top-6 | Search-R1 冻结 E5 top-3 |
| 语料 | 数据集对应 Wikimedia corpus | Search-R1 Wiki-2018 |
| 状态表示 | 句子级 Evidence Context | Search-R1 当前可见 conversation |
| 查询策略 | gap-guided query | Search-R1 原生 search query |

因此，本实验可以声称 **S2G-code-aligned question set** 和
**S2G-compatible structured supervision**，不能声称严格复现 S2G-RAG
表格中的绝对 EM，也不能仅凭绝对 EM 与其 `43.3` 横向定胜负。

### 2.3 论文文字与公开代码的歧义

论文正文写“official development split”，公开 evaluator 则保留预测文件
中前 1,000 个唯一问题 ID。本协议优先遵循可执行、可哈希的公开代码口径，
并在论文中明确称为 `S2G-code-aligned HotpotQA-dev-1000`，不将其写成
未经作者确认的“论文完整测试集”。

## 3. 数据划分

### 3.1 现有种子数据

现有 S2G 最小验证实际包含：

- 100 个独立问题；
- 246 个训练状态；
- 143 个原动作标签 STOP；
- 103 个原动作标签 CONTINUE。

旧文档中的 `seed_train_200` 是历史命名，不代表 200 个训练问题。本协议
统一改称 `existing_seed_100q_246s`。

现有种子只用于链路复用和少量起始监督：

- 不作为 validation；
- 不进入最终测试；
- 在扩充版训练中的状态权重占比原则上不超过 10%；
- 若与新 HotpotQA train 问题发生规范化文本重复，优先排除旧种子。

在 full Teacher 标签生成前，进一步以
`s2g-scaleup-seed-use-amendment-20260728.json` 冻结为：扩充版主
checkpoint **不混入**这 246 个旧状态。原因是这些问题已经作为开发数据
反复查看，且其 Teacher 生成协议与本轮不完全同源。旧 LoRA 只保留为
secondary historical comparator；若 fresh 数据未达到 Gate S3，只能按
预注册规则解封 `training_reserve_200`，不能事后回填旧 seed。

### 3.2 新增训练问题

从 HotpotQA `distractor/train` 中确定性、标签盲选择：

| Split | 问题数 | 用途 |
|---|---:|---|
| `fresh_train_700` | 700 | 正式训练 |
| `grouped_validation_100` | 100 | checkpoint 与训练诊断 |
| `training_reserve_200` | 200 | 仅在数据门失败时解封 |

选择前排除：

- 与 HotpotQA validation 问题 ID 或规范化题面重复；
- 现有 Core200、Gate H0、Gate A/B/C 和 legacy/pilot 使用过的题面；
- train、validation、reserve 之间的 ID 或规范化题面重复。

排序规则：

```text
rank = SHA256(
    seed
    + NUL
    + original_hotpot_id
    + NUL
    + normalized_question
)
```

冻结 seed：

```text
fresh_train_700:
  20260727-s2g-scaleup-fresh-train-700-v1
grouped_validation_100:
  20260727-s2g-scaleup-grouped-validation-100-v1
training_reserve_200:
  20260727-s2g-scaleup-training-reserve-200-v1
```

所有训练/验证划分以 `question_id` 为分组单位。禁止同一问题的不同轮次
跨 split；这比原论文按 snapshot 随机 90/10 划分更严格。

### 3.3 最终评测问题

从 HotpotQA `distractor/validation` 原始顺序物化前 1,000 个唯一
原始 HotpotQA ID：

```text
s2g_code_aligned_hotpotqa_dev_1000
```

要求：

- 保存原始 ID、题面、原始行号和 SHA256；
- 不按 Search-R1 内部 `dev_i` 排序；
- 在 checkpoint、prompt、阈值与失败回退全部冻结前，不运行端到端评分；
- Core200 即使与这 1,000 题有交集，也不参与模型选择；交集必须单独
  报告。为了得到更干净的确认性证据，主报告同时提供排除已查看交集后的
  sensitivity analysis。

## 4. 轨迹与 Teacher 监督

### 4.1 状态采集

- 冻结 Search-R1 7B、E5 top-3、Wiki-2018、prompt、最大 4 轮和解码；
- 每个新训练问题记录最多 4 个 turn-level 当前可见状态；
- 这里的 4 个状态是初始状态与前 3 次搜索后的状态（state 0–3）；
  若 Native 执行第 4 次搜索，第 4 次搜索后的预算耗尽终态不属于
  STOP-vs-CONTINUE 决策，因此不进入 Teacher、训练或 STOP 指标；
- 主训练优先使用 Native 可达状态；
- forced-continue / counterfactual 状态必须带显式标记，且在主训练中的
  比例不超过 25%；
- 禁止把 gold answer、未来 query、未来文档、未来动作、官方 EM 写入
  Judge 输入。

### 4.2 Teacher

- Teacher：本地 `Qwen3.6-35B-A3B`
- temperature：0
- thinking：false
- 最大输出：512 tokens
- 输出：严格 `sufficient + gap_items` JSON
- 仅允许读取 question 与当前 context
- 不允许读取 gold answer、未来状态或 answer-probe correctness

### 4.3 清洗

依次执行：

1. JSON schema 与字段枚举检查；
2. `sufficient=true` 时 `gap_items=[]`；
3. `sufficient=false` 时至少一个具体 gap；
4. context-only 人工抽样审计；
5. 使用 gold supporting-title coverage 仅作冲突过滤，绝不写入模型输入；
6. 按 question、turn、Teacher 标签和状态来源报告分布；
7. 不通过删除难例强行制造完美平衡。

## 5. 数据规模门

不以“选了多少问题”代替有效训练量。进入正式 LoRA 前必须同时满足：

- 清洗后 train snapshots `>= 2,500`；
- grouped validation snapshots `>= 280`；
- train 独立问题 `>= 600`；
- sufficient / insufficient 原始自然分布中的少数类占比 `>= 15%`；
- turn 1–4 每一轮占比 `>= 10%`；
- schema parse rate `>= 98%`；
- question-level split overlap 为 0；
- 输入泄漏审计为 0。

若只因训练状态数或轮次覆盖未达门，才允许通过 append-only amendment
解封 `training_reserve_200`。类别占比失败必须如实停止，不得靠 reserve
定向挑题。不得因为 validation 指标不好而临时增补特定类型问题。

> **2026-07-27 预标签修订**：现有 246-state seed 已知只有
> 44/246=17.9% `sufficient`。原 35% raw-class 门会诱导事后删难例，
> 因而由
> `s2g-scaleup-class-balance-amendment-20260727.json` 覆盖为 15%。
> 原始 train、grouped validation 和 final 均保留自然分布；只在每个
> train epoch 内确定性过采样少数类，使两类有效训练贡献相等。reserve
> 不再因类别比例失败而解封。

### 5.1 训练轨迹分片修订

Gate S2 的 100 题是 `fresh_train_700` 冻结顺序的精确前缀。为避免
重复运行相同的 7B 轨迹，
`s2g-scaleup-train-sharding-amendment-20260728.json` 在 smoke
完成前冻结以下执行方式：

- `smoke100 = rows[0:100]`；
- `remainder600 = rows[100:700]`；
- 两个容器运行分别完成 exit、OOM、顺序、错误和无标签隔离审计；
- 状态与 Teacher 结果也分别审计；只在 student rows 层按
  `smoke100 + remainder600` 拼接；
- 禁止伪造一个合并后的 `run_start/run_end`。

两片题目顺序拼接后与原 `fresh_train_700` 的 700 个冻结问题逐项一致，
跨片问题重叠为 0。该修订只减少重复计算，不改变模型、检索器、prompt、
解码或搜索预算。

### 5.2 清洗规则修订

`s2g-scaleup-cleaning-amendment-20260728.json` 在任何扩充 Teacher 标签
产生前冻结清洗规则：

- 自动删除仅限 generation error 或严格 schema invalid，且保留率必须
  达到 98%；
- Teacher 全量无标签运行完成后，才允许隔离读取训练 gold supporting
  facts；
- 只有当当前 context 经确定性规范化后包含全部官方 gold
  supporting-fact 句子、但 Teacher 仍判 `insufficient` 时，才视为强
  冲突并从训练排除；
- 仅标题覆盖不能触发排除；Teacher 判 sufficient 但未覆盖全部 gold
  句子也保留，因为可能存在替代证据；
- 冲突项只排除并报告，不做人工改标。

## 6. 训练

- Base：冻结 revision 的 `Qwen3.5-2B`
- 监督目标：完整结构化 JSON 的 token-level autoregressive NLL
- LoRA：`r=16`、`alpha=32`、`dropout=0.05`
- target modules：与既有 S2G 最小公平 LoRA 完全一致，即
  `q_proj/k_proj/v_proj/o_proj/in_proj_a/in_proj_b/in_proj_qkv/`
  `in_proj_z/out_proj/gate_proj/up_proj/down_proj`
- epochs：3
- learning rate：`1e-4`
- max length：3072（原冻结值 2048；经 2026-07-29 全量、标签盲
  prompt-length 审计和显式 amendment 修订；0 删除、0 截断）
- gradient accumulation：8
- checkpoint 选择：只使用 grouped validation 的自然分布
  **Teacher-target token NLL**，取最低值，平局取最早 epoch；此时不读取
  gold answer、SAFE_EARLY_STOP、STOP precision/recall/AP 或端到端 EM
- 每个 epoch 对少数 Teacher 类确定性过采样到多数类数量；
- grouped validation 保持自然分布，不做过采样或下采样；
- Core200、S2G-code-aligned dev-1000 均不得选择 checkpoint

必须保存训练曲线、每个 checkpoint 的 hash、训练/验证 question ID hash、
模型 revision、tokenizer、prompt、容器与运行时版本。

## 7. 指标

### 7.1 三种 STOP 真值

| 真值 | 含义 | 作用 |
|---|---|---|
| `TEACHER_SUFFICIENT` | Teacher 判断 context 足够 | 衡量 Teacher 模仿 |
| `RETRIEVAL_COMPLETE` | 已检索标题覆盖全部 gold supporting titles | 对齐 S2G 机制分析 |
| `SAFE_EARLY_STOP` | 当前标准化答案正确，且 Native 下一步原本仍会 SEARCH | 本论文主要停止能力指标 |

Teacher 标签的 precision/recall 不能单独证明系统学会了安全提前停止。

### 7.2 状态级主要指标

在 grouped validation 和最终评测中报告：

- `SAFE_EARLY_STOP precision`
- `SAFE_EARLY_STOP recall`
- `SAFE_EARLY_STOP Average Precision`
- 在匹配 precision 下的 recall
- premature STOP 数与比例
- confusion matrix
- score calibration / reliability

若结构化 Judge 只输出 boolean，必须额外记录
`log P(sufficient=true) - log P(sufficient=false)` 或等价的固定位置
log-prob margin，才能形成 PR 曲线；贪心 boolean 只提供一个 operating
point，不能替代 AP。

`s2g-scaleup-score-margin-amendment-20260728.json` 已在扩充 Teacher
标签生成前把该口径具体冻结为：

```text
canonical JSON key order = sufficient, gap_items
score prefix = {"sufficient":
score = log P(true | prompt + prefix) - log P(false | prompt + prefix)
```

这里仍使用同一个结构化 Judge prompt，并非额外 answer probe。贪心完整
JSON 用于 parse/gap 质量与默认 operating point；固定前缀 margin 用于
PR、AP、matched-precision recall 和校准。阈值只能在 grouped validation
选择，并在连接 dev-1000 labels 前冻结。

### 7.3 问题级提前停止指标

- `early-stop opportunity recall`：存在安全提前停止机会的问题中，实际
  被安全提前停止的比例；
- `first-safe-stop delay`：策略停止轮次相对最早安全停止轮次的差；
- 每题避免的搜索次数；
- 安全停止题上的平均搜索节省；
- 因过早停止导致的正确转错误题数；
- 因避免后续干扰导致的错误转正确题数。

### 7.4 系统级硬指标

- Official EM
- Token F1
- 平均搜索次数
- 搜索次数分布
- 平均 Judge 调用次数与延迟
- 每题总推理时间

主要系统成功条件：

```text
Official EM 相对 Native 的配对差异达到预注册非劣界；
并且平均搜索次数显著下降。
```

STOP precision/recall 是“是否学到停止判别”的主证据，但不能替代上述
端到端硬门。

## 8. 对照与统计

同题比较：

1. Native Search-R1
2. Structured Base Qwen3.5-2B
3. 现有 246-state S2G LoRA
4. 扩充训练后的 S2G LoRA
5. 固定搜索预算策略

至少报告：

- Base vs expanded LoRA 的状态级配对 bootstrap CI；
- Native vs expanded LoRA 的 question-level EM 与搜索次数配对 CI；
- McNemar 正确性转移表；
- Bridge / Comparison 分层；
- Native 搜索轮次分层；
- Core200 结果只列为历史开发结果，不与新 final 结果合并。

最终非劣界、bootstrap seed 和多重比较处理必须在连接最终标签前通过
append-only statistical amendment 冻结。

### 8.1 统计与阈值预注册修订

`s2g-scaleup-statistical-amendment-20260728.json` 已在 grouped
validation 标签和输出均未查看时冻结：

- 每个 Judge 仅在 grouped validation 选择各自 margin 阈值；
- 在至少预测 10 个 STOP 的条件下，要求经验 STOP precision
  `>= 0.90`，随后最大化 recall；并列时依次选择 precision 更高、
  threshold 更高者；
- 若没有可行阈值，冻结为 `never early STOP`，不得用 final 重新选；
- 主要状态比较为 expanded LoRA 相对 Structured Base 的
  `SAFE_EARLY_STOP AP`；
- 主要系统比较为 expanded LoRA 相对 Native 的官方 EM 与平均搜索数；
- paired question bootstrap 为 10,000 次，seed `20260728`；
- EM 非劣界为绝对 `-2pp`，要求 expanded-minus-Native 的 95% CI 下界
  大于 `-0.02`；搜索减少要求搜索次数差的 95% CI 上界小于 0；
- 旧 246-state LoRA 与固定预算策略为 secondary/descriptive，不共享
  主要结论。

## 9. Gate

### Gate S0：协议与来源冻结

- 本文档和机器可读 protocol 均已生成；
- S2G 官方 commit、数据 revision、split seed 和排除规则已冻结；
- 训练与最终评测原始 ID 可重放。

### Gate S1：划分与无泄漏审计

- 700/100/200 问题级划分完成；
- S2G-code-aligned dev-1000 入口完成；
- 所有交集与重复审计通过；
- 未连接最终评测输出。

### Gate S2：100 题状态采集 smoke

- 100/100 题完成；
- 0 trajectory error；
- 0 输入泄漏；
- 状态、轮次与顺序审计通过；
- 不因模型效果差而修改已冻结运行设置。

### Gate S3：论文量级监督数据

- 全量状态采集、Teacher 标注与清洗完成；
- 第 5 节数据规模门全部通过；
- 若失败，只能按预注册条件解封 reserve 或如实停止。

### Gate S4：LoRA 与 grouped validation

- 3 epochs，loss finite，0 OOM；
- Base、旧 LoRA、新 LoRA 同协议评测；
- checkpoint 仅由 grouped validation 的自然分布 Teacher-target token NLL
  冻结，不使用随后开启的答案标签或 STOP/端到端指标；
- STOP precision/recall/AP 与端到端开发指标完整。

### Gate S5：S2G-code-aligned dev-1000

- checkpoint、prompt、阈值和统计协议已冻结；
- Native、Base、旧 LoRA、新 LoRA 同题完成；
- 结构审计后一次性评分；
- 无论改善、持平或失败都生成报告。

## 10. 时间预算

按本机已有实测粗估：

| 阶段 | 预计耗时 |
|---|---:|
| 数据物化与审计 | 1–3 小时 |
| 700 train + 100 validation 轨迹 | 18–24 小时 |
| 约 3,000 状态 Teacher 标注 | 1–2 小时 |
| LoRA 3 epochs | 3–4 小时 |
| grouped validation | 1–3 小时 |
| S2G-code-aligned dev-1000 Native 轨迹 | 24–30 小时 |
| Judge 离线评分与报告 | 4–8 小时 |

总墙钟时间约 2–3 天。Search-R1 7B 轨迹采集是主要瓶颈。

## 11. 当前执行状态

- [x] 研究问题与对齐边界冻结
- [x] STOP precision/recall 的三种真值口径冻结
- [x] 训练规模门冻结
- [x] 类别平衡预标签 amendment 冻结
- [x] 训练 100+600 分片 amendment 冻结
- [x] Teacher 清洗 amendment 冻结
- [x] 旧 246-state seed 排除规则冻结
- [x] state 0–3 决策域 amendment 冻结
- [x] sealed reserve 条件执行 amendment 冻结
- [x] STOP margin 序列化与评分 amendment 冻结
- [x] 阈值与统计 amendment 冻结
- [x] 机器可读 protocol
- [x] HotpotQA train / validation 数据来源审计
- [x] 700/100/200 split manifest
- [x] S2G-code-aligned dev-1000 manifest
- [x] 执行代码与全部 amendment 的 31 文件哈希清单
- [x] reserve 后续训练/验证/final 隔离链 amendment 与 v2 哈希清单
- [x] 100 题状态采集 smoke（100/100，0 error，exit 0，0 OOM）
- [x] Gate S2 原始审计 FAIL 证据保留
- [x] Gate S2 smoke 子集审计 compat-v2 PASS（无 Teacher 重跑）
- [x] 全量状态采集（700 train + 100 grouped validation）
- [x] Teacher 标注与清洗
- [x] reserve200 补量与 Gate S3 复审（900 questions / 3,009 states）
- [x] training `gate-s3` 软链接挂载失败证据与只读直挂 recovery amendment
- [x] 扩充版 LoRA（3 epochs / 12,654 steps / epoch-001 冻结）
- [x] grouped validation 冻结与 Gate S4（阈值仅由 grouped 选择）
- [ ] dev-1000 同题最终评测（Native 标签盲轨迹运行中）

### 2026-07-27 Gate S0/S1 审计结果

- 官方 train：90,447 行，566,426,227 bytes，SHA256
  `26650cf50234ef5fb2e664ed70bbecdfd87815e6bffc257e068efea5cf7cd316`；
- 官方 distractor validation：7,405 行，46,320,117 bytes，SHA256
  `4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea`；
- 官方 train 中有 8 个重复规范化题面。源数据保持不变；选择器保证同一
  规范化题面最多进入一个 split；
- 远端历史结果中抽取到 2,561 个已使用题面，全部作为无标签排除项；
- `fresh_train_700`、`grouped_validation_100`、
  `training_reserve_200`、`dev_1000` 均达到冻结题数，ID 与规范化题面
  两两零重叠，盲运行 inputs 中 gold 字段为 0；
- 当前 Search-R1 `test.parquet` 对官方题面执行了
  `strip + 缺失问号补齐`。dev-1000 有 87 条字面变化，但按该确定性
  规范化后 1,000/1,000 顺序完全一致。正式入口保留官方原始题面和
  原始 HotpotQA ID，不再使用内部 `dev_i` 作为身份；
- labels 只保留在远端隔离目录，不同步到本地；本地仅同步 manifest 与
  无标签 inputs。

### 2026-07-28 Gate S2 运行状态

- smoke 容器 `searchr1-v02-s2g-scaleup-gate-s2-smoke100` 已自然结束：
  `run_start=1`、`trajectory=100`、`run_end=1`、0 trajectory error、
  exit 0、`OOMKilled=false`；
- 结构审计 PASS，100 个唯一问题按冻结顺序完全一致；状态审计 PASS，
  得到 345 个 state 0–3 状态，其中 Native 下一步 SEARCH 262、
  ANSWER 83，并排除了 17 个第四次检索后的预算耗尽终态；
- 40-state Qwen3.6 Teacher smoke 本身 40/40 生成成功、parse rate 1.0、
  clean 40、0 schema error、0 labels。原始 v1 审计仍判 FAIL，因为
  full-run 审计器硬编码要求 `source_count == input_count`，而 smoke
  按协议是 345 source 的冻结 40-row 子集；
- 原始 Gate S2 completion status 和 v1 audit FAIL 均永久保留，不覆写。
  在任何 downstream 运行或 label 开启前冻结了 ordered-subset
  compatibility amendment，仅对既有 Teacher run 做确定性重审：
  40 个 ID 是 345 source 的有序唯一子集，result/input 顺序、
  sequence/source-sequence、question/state 元数据与模型字段投影全部
  一致。compat-v2 PASS，没有重新调用 Teacher；
- `decision-compat-v2.json` 已解除轨迹链阻塞；剩余 600 train 与
  grouped validation 100 的标签盲轨迹均已自然结束并通过独立审计：
  两片分别为 600/600、100/100，均 `run_start=1`、`run_end=1`、
  0 trajectory error、exit 0、`OOMKilled=false`。状态审计分别得到
  2,028 与 331 个 state 0–3 状态，均 PASS；
- 完整 Teacher 监督器已在上述两片完整通过后按协议启动三片 full
  Teacher 标注。`fresh_train_prefix_100` 的 345 个请求已全部生成，
  343 条 clean，parse rate 99.42%，2 条 top-level schema error 按冻结
  规则排除，审计 PASS；`fresh_train_remainder_600` 的 2,028 个请求
  正在运行。任一科学运行失败仍保存证据并停止，不自动重试；
- 截止本次状态审计，两个监督器均声明
  `labels_mounted=false`、`gold_fields_received=0`。
- 训练、grouped validation、final dev-1000 监督器均已处于只读等待态；
  它们不会越过前置 Gate，也不会在盲推理完成前读取对应 labels。
- 因 state 0–3 无标签投影预计 700 题约 2,411 个状态，低于冻结的
  2,500 门槛，已在任何 reserve label 开启前预注册独立 reserve
  后续链。它只在 `gate-s3-reserve/decision.json` 明确 PASS 后，将
  `gate-s3-reserve` 映射为隔离仓库视图的 `gate-s3`，复用完全相同的
  训练、grouped validation 与 final 代码；原始失败证据和结果目录均不
  覆盖，门槛、checkpoint、阈值与统计规则均不变。
- reserve 链第一次空等待启动时，代码清单中的 grouped-validation
  SHA 因人工录入少 4 个字符而校验失败，但复合 shell 未 fail-fast，
  导致等待进程被拉起。发现时 reserve 仍封存、shadow repo 未创建、
  0 子任务和 0 labels 开启；已终止该空等待进程并完整归档证据。纠正
  后的 v2 清单使用 fail-fast 复算 6/6 文件、0 mismatch，新的监督器
  仅处于 `waiting_for_reserve`。这不是科学重试。
- 31 个冻结协议、amendment 与执行文件已在本地和远端逐文件复算，
  0 hash mismatch。清单 SHA256 为
  `f8b471e439aea83de78727f23ea0a6c578c1eb7aeb1bca32bb197f5c800b0716`。
  正式轨迹 runner 明确记录为远端独立冻结路径
  `scripts/run_searchr1_v02_smoke.scaleup-v2.py`，不覆盖仓库中的旧通用
  runner。
- Gate S2 compat 关键远端 SHA：
  - 原 completion FAIL：
    `909184b9a0524ea8852035af19d354f80b828ce447eb834ceb4552efb43e9bd9`
  - 原 Teacher audit FAIL：
    `a4e61d609a2527e2e85618666b97436a12f7c3e104c801a24b4820fe157c2d1d`
  - 未变 Teacher run：
    `2f40f975e2bd9f3142494f92d041af23eb423e2262a45710ad6e861bc9d48e2f`
  - compat-v2 audit PASS：
    `fed5b6ab6317809ae9c8326deb2486f5c8fd947e09f5158df39b527c2a48b4ba`
  - compat-v2 decision：
    `9a2ff85a324160d3ddb475274b5a970f091951c176927af550d80a14545b4e9b`

可复核入口：

- 机器协议：
  `deploy/searchr1-v02/s2g-scaleup-aligned-eval-protocol-20260727.json`
- 执行代码清单：
  `deploy/searchr1-v02/s2g-scaleup-execution-code-manifest-20260728.json`
- reserve 链 amendment：
  `deploy/searchr1-v02/s2g-scaleup-reserve-chain-amendment-20260728.json`
- reserve 链纠正后代码清单：
  `deploy/searchr1-v02/s2g-scaleup-reserve-chain-code-manifest-v2-20260728.json`
- 预激活启动偏差：
  `deploy/searchr1-v02/s2g-scaleup-reserve-chain-launch-deviation-20260728.json`
- Gate S2 子集审计 amendment：
  `deploy/searchr1-v02/s2g-scaleup-gate-s2-subset-audit-compatibility-amendment-20260728.json`
- Gate S2 子集审计代码清单：
  `deploy/searchr1-v02/s2g-scaleup-gate-s2-subset-audit-code-manifest-20260728.json`
- 物化器：`scripts/prepare_s2g_scaleup_splits.py`
- 本地证据：
  `results/searchr1-v02-reproduction/s2g-scaleup-aligned-eval-v1/`
- 远端冻结入口：
  `evaluation/s2g-scaleup-frozen-v1/`

### 2026-07-28 Gate S3 Teacher 运行状态

- 700 train 的状态总数为 `345 + 2,028 = 2,373`；grouped validation
  为 331，总计 2,704 个标签盲 Teacher 请求；
- 轨迹监督器完成时仍声明 `labels_mounted=false`、
  `gold_fields_received=0`；所有 Teacher 输入的模型可见字段严格为
  `question` 与 `context`；
- 首个 345-state 分片用 Qwen3.6-35B-A3B、concurrency 4 完成，0
  generation error、0 JSONL 结构错误；审计后 343 条进入 student
  输出；
- 主训练 2,028-state 分片正在运行。只有三片 Teacher 结果全部完成并
  通过审计后，Gate S3 才允许开启训练 labels 做单向 strong-conflict
  过滤和冻结训练集。

### 2026-07-29 Teacher 恢复、Gate S3 与 reserve200

- 远端主机重启导致原 Teacher 主分片容器以 `exit 255` 终止；原输出、
  容器 inspect、状态和日志均保留，未静默重试。第一次获批恢复又在
  任何 Teacher 请求发出前遇到重启自启动 NAS LLM 占用
  `127.0.0.1:8000`，容器保持 `created`、`exit 128`、非 OOM；该次
  失败也完整保留。
- 经用户明确批准后，使用新容器名和新输出目录重启，临时停止端口与
  GPU 竞争服务，并在终态恢复服务。恢复 Teacher 最终 `exit 0`：
  - `fresh_train_remainder_600`：2,028/2,028 返回，2,017 clean，
    parse rate 99.46%，11 条 top-level schema error；
  - `grouped_validation_100`：331/331 返回，331 clean，parse rate
    100%；
  - Teacher supervisor 与外层恢复 wrapper 均为 `complete`，
    `labels_mounted=false`、`gold_fields_received=0`。
- Gate S3 通过隔离 shadow repo 将上述恢复产物映射到原冻结文件名，
  没有覆盖任何原失败产物；随后才开启 train labels。单向 strong
  conflict 过滤从 prefix 343 条中排除 2 条、从 remainder 2,017 条中
  排除 8 条，最终得到 2,350 个 train 状态、700 个问题。
- Gate S3 的问题隔离、类别占比、轮次占比、请求 ID 唯一性以及
  grouped validation 331-state 门均 PASS；唯一失败项是
  `train_states >= 2500`。因此终态为
  `data_gate_failed_reserve_required`，不是基础设施失败，也没有降低
  门槛。
- 已按预注册固定顺序启动 `training_reserve_200`。截至启动审计，
  supervisor 为 `reserve_trajectory_running`，容器运行、0
  trajectory error、非 OOM；reserve labels 仍封存。reserve 完成
  Teacher 盲标注和审计后，才允许开启它自己的 train-only labels 做
  同一 strong-conflict 过滤。
- reserve 运行期间出现一次约 6 分钟的 Tailscale 控制面中断。恢复后
  boot ID 未变，trajectory 从 4 增长到 8，0 error，证明 detached
  科学运行持续执行、没有重跑。恢复专用 downstream waiting guard 已
  启动为 `waiting_for_reserve`；它在 reserve PASS 前不会创建训练
  shadow repo 或开启下游 labels。

关键证据 SHA256：

- 恢复 Teacher status：
  `4eb78a70f0a1525e2dd540a727df54dd5563563a64c6454806cc253f53d2ba28`
- 恢复 wrapper status：
  `9d6fe5a1b0a113ef34f7d97cb5b68cd28e6d732595987f1c27cc11508381defd`
- 恢复主分片 student：
  `d4ba12403bf70f49f4448648651c68681d18e4a65b01077fdd552a23e337f288`
- 恢复 grouped validation student：
  `5fc3c02d8e117b271bc4082e54d2ccdc33fc413f614a4d9f3f132d8031643fa3`
- 恢复 Gate S3 status：
  `9fdc0f76cc33e78bea29f2a89a1cdcd8597fbdf6b0a704fb5932e351d53f0505`

恢复与 reserve 的可复核入口：

- Teacher 主机重启 incident：
  `results/s2g-scaleup-aligned-eval-v1/monitoring-incidents/20260729T102251+0800-teacher-host-reboot.md`
- Teacher 端口冲突 incident：
  `results/s2g-scaleup-aligned-eval-v1/monitoring-incidents/20260729T104139+0800-teacher-recovery-port-conflict.md`
- 恢复 Gate S3 mapping amendment：
  `deploy/searchr1-v02/s2g-scaleup-recovered-gate-s3-mapping-amendment-20260729.json`
- 恢复 Gate S3 纠正后 manifest：
  `deploy/searchr1-v02/s2g-scaleup-recovered-gate-s3-code-manifest-v2-20260729.json`
- reserve service guard amendment：
  `deploy/searchr1-v02/s2g-scaleup-recovered-reserve-service-guard-amendment-20260729.json`

### 2026-07-29 v2 reserve recovery 与训练挂载恢复

- v2 recovery 在不重采 200 条 trajectory 的前提下，预检精确
  `kstar.searchr1_v02_run_audit` 子模块并完成全部标签盲审计。
- reserve 产生 670 个状态；Teacher clean 663，排除 4 个 strong
  conflicts，保留 659。合并后的冻结 train 为 900 个问题、3,009
  states，insufficient / sufficient = 2,109 / 900；Gate S3 全门 PASS。
- 第一次下游 training smoke 因 `/run/gate-s3` 宿主软链接目标未进入
  容器 namespace 而 `exit 1`，非 OOM。训练尚未开始，grouped/final
  labels 未打开，原失败现场完整保留。
- append-only recovery 只把同一冻结 Gate S3 目录显式只读挂载至
  `/run/gate-s3`，并使用不同容器名和影子输出目录；数据、prompt、
  LoRA 超参数、选择规则和评测协议不变。相关 19 项测试通过，恢复
  smoke 最终 `exit 0`、非 OOM、loss finite。
- full training 的全量 dataset 预检发现 36 个 train 和 2 个
  validation state-3 状态超过 2048，最大 2,662；尚未执行任何 full
  training step。长度审计不输出标签、不计算科学指标。
- 为保留晚轮监督且避免截断证据，训练 `max_length` 显式修订为 3072。
  train/validation 全部行及哈希不变，0 删除、0 截断；prompt、target、
  其余训练参数、选模和评测协议不变。新恢复链的 smoke 已 PASS，
  3-epoch full training 正在运行。
- amendment：
  `deploy/searchr1-v02/s2g-scaleup-training-gate-symlink-mount-recovery-amendment-20260729.json`
- incident：
  `results/s2g-scaleup-aligned-eval-v1/monitoring-incidents/20260729T211147+0800-training-gate-symlink-mount-failure.json`
- max-length amendment：
  `deploy/searchr1-v02/s2g-scaleup-training-max-length-3072-recovery-amendment-20260729.json`
- overlength incident：
  `results/s2g-scaleup-aligned-eval-v1/monitoring-incidents/20260729T212131+0800-full-training-prompt-overlength.json`
- 下游只读预检在 full training 期间发现活动 shadow 缺少顶层 `kstar`，
  且 grouped/final 容器仍会跨父 bind mount 访问宿主软链接。原
  supervisor 已逐字节备份；修订版只增加顶层包链接和显式只读
  `gate-s3` / `grouped-validation100` 子挂载，远端 import/mount
  预检 PASS，不改训练、prompt、阈值或评测协议。
- downstream preflight amendment：
  `deploy/searchr1-v02/s2g-scaleup-downstream-mount-preflight-amendment-20260729.json`
- downstream preflight status：
  `results/s2g-scaleup-aligned-eval-v1/downstream-mount-preflight-status.json`
- 第一次真实 Docker smoke 继续暴露 grouped 目录内部
  `state-manifest.jsonl` / `trajectories.jsonl` 为跨父 bind mount 的绝对
  软链接，因而 `exit 3`。该失败发生在 grouped 启动及 labels 开启前。
  追加纯基础设施 amendment 后，supervisor 从实际 state manifest
  解析权威根目录并显式只读挂载；原 mount-safe 版本逐字节备份，第二次
  真实容器 smoke `exit 0`，不改科学配置。
- grouped inner-symlink amendment：
  `deploy/searchr1-v02/s2g-scaleup-grouped-inner-symlink-recovery-amendment-20260729.json`
- grouped inner-symlink recovery status：
  `results/s2g-scaleup-aligned-eval-v1/grouped-inner-symlink-recovery-status.json`
- final dev1000 的标签盲输入、runner 和顶层 `kstar` 已在真实 Docker
  namespace 完成独立可见性、哈希、导入和 `--help` 预检；全程未挂载或
  打开 final labels。
- final input/runner preflight status：
  `results/s2g-scaleup-aligned-eval-v1/final-dev1000-input-runner-preflight-status.json`
- final 中文 Material Passport 在 labels 封存、扩充 LoRA 仍训练时补全为
  同题四行表：Native、Structured Base、旧 S2G LoRA（历史比较）和扩充
  S2G LoRA；机器可读指标、推理、阈值、统计与决定规则均未改变。
- final report completeness amendment：
  `deploy/searchr1-v02/s2g-scaleup-final-report-completeness-amendment-20260729.json`
- final report completeness code manifest：
  `deploy/searchr1-v02/s2g-scaleup-final-report-completeness-code-manifest-20260729.json`

### 2026-07-30 完整训练与 grouped pre-blind recovery

- 扩充版 Qwen3.5-2B S2G LoRA 已完整训练 3 epochs、12,654 steps；
  `all_losses_finite=true`、`container_oom_killed=false`。按冻结的自然
  grouped-validation Teacher-target token NLL 规则选择
  `epoch-001`，adapter tree SHA256 为
  `cfed584b95025cc4f1fd555cd2236b2773f1289c67dab03d26b2829659578767`。
  final dev1000 未用于选模。
- 训练结束后，grouped 在科学输出和标签开启前依次暴露三类问题：
  1. Reasoner/Retriever 重启后的服务就绪竞态；
  2. fresh shadow 把 partial `kstar` overlay 当成完整运行包；
  3. 普通 Native Search-R1 轨迹没有 always-reject
     `candidate_events`，而 legacy Gate-H0 builder 默认只接受后者。
- 三个失败目录、status 与 stderr 均 append-only 保留；每次失败时
  grouped/final labels 均未开启，没有 token-state manifest、Judge
  预测或科学指标，也没有 OOM 或训练重跑。
- v4 recovery 只做工程/协议兼容：
  - grouped 与 final 前要求 Reasoner/Retriever 连续两次 HTTP 200；
  - 物化 filtered runtime package base，并叠加当前审计 overlay；
  - Gate-H0 默认继续 fail closed，仅 grouped 显式传入兼容开关时，
    对完全缺失 `candidate_events` 的普通 Native 轨迹令
    `native_search_count=search_calls`；不重写 trajectory。
- v4 已通过 23 项定向测试和全仓 660 项测试，并越过三处旧失败点。
  当前 245-state answer-only probe 正在标签盲运行；grouped/final
  labels 仍封存，不报告中间 STOP、EM 或搜索次数指标。
- 机器可读 incident：
  `results/s2g-scaleup-aligned-eval-v1/monitoring-incidents/20260730T053300+0800-grouped-preblind-recovery-chain.json`
- v2/v3/v4 amendments：
  - `deploy/searchr1-v02/s2g-scaleup-grouped-service-readiness-recovery-amendment-20260730.json`
  - `deploy/searchr1-v02/s2g-scaleup-grouped-runtime-package-recovery-amendment-20260730.json`
  - `deploy/searchr1-v02/s2g-scaleup-grouped-native-schema-recovery-amendment-20260730.json`
- 在 final dev1000 尚未生成 trajectory/status/output 且 final labels
  封存时，前瞻审查确认 final token-state builder 需要同一显式 Native
  schema 开关。旧 supervisor SHA256
  `a75cbd53a58ee433d065537122da1700b1ddd6251ebdf27d10ab67d637e7c5ee`
  已逐字节备份；兼容版 SHA256
  `6e260011b6ba562d005ad08b4de82da230af6daf0f8234ffcf77572376144cda`
  已通过 17 项相关测试和远端 preflight。此修订只 supersede 前一
  amendment 的 final supervisor code hash，不改任何科学设置：
  `deploy/searchr1-v02/s2g-scaleup-final-native-schema-compatibility-amendment-20260730.json`。
- grouped answer probes 运行期间、三组 Judge 尚未启动且 grouped/final
  labels 仍封存时，前瞻报告审查发现阈值文件以 `sort_keys=True`
  序列化后，旧评测器会把字母序第一个 `expanded_s2g_lora` 错当
  pairwise reference；而冻结的 final supervisor 需要
  `expanded_s2g_lora_minus_structured_base`。旧实现会在 final 标签
  已开启、评测已完成后才以 `KeyError` 失败。修复仅把存在
  `structured_base` 时的 pairwise reference 显式固定为该基线，保持
  bootstrap 10,000 次、seed、阈值和所有科学设置不变。旧/新 evaluator
  SHA256 分别为
  `ca3fe3f005196899b3fae26268decab7ca81319c2587cbf515e515108a4f735c`
  与
  `88558b8c3ba76e03d099a69dd139aecd46ea5d6d38263222fb0323c8612f090c`，
  16 项相关测试 PASS。
- 初次部署在 source 更新后对 shadow 再次 `mv` 时失败，因为活动
  shadow 的 `scripts` 实为 source `scripts` 的软链接；失败现场、
  stderr 和文件哈希已保留。symlink-aware recovery 没有再次写入
  evaluator，只验证两个 canonical path 指向同一新文件，并补齐
  amendment 与部署审计。部署快照为 probes 161/245、0 Judge output、
  0 grouped evaluation、grouped/final labels 封存、0 实验重启：
  - `deploy/searchr1-v02/s2g-scaleup-pairwise-reference-recovery-amendment-20260730.json`
  - `deploy/searchr1-v02/s2g-scaleup-pairwise-reference-shared-scripts-recovery-amendment-20260730.json`
  - `results/s2g-scaleup-aligned-eval-v1/pairwise-reference-recovery-deployment-audit.json`
  - `results/s2g-scaleup-aligned-eval-v1/pairwise-reference-recovery-test-audit.json`
  - `results/s2g-scaleup-aligned-eval-v1/monitoring-incidents/20260730T072035+0800-pairwise-reference-deployment-shared-scripts.json`

### 2026-07-30 answer-only 服务协议执行恢复

- v4 最终完成了 245/245 个标签盲 probe，顺序、状态 ID、输入哈希、
  0 probe error 等工程审计均正确，但结构审计如实 FAIL：
  `ANSWER/SEARCH/INVALID=15/228/2`。原因不是 answer-only probe 的科学
  结果，而是 supervisor 把冻结状态发送到普通 Native Search-R1 服务；
  该服务的 `STOP_AFTER_SEARCHR1_ACTION=false` 且
  `CONSTRAIN_SEARCHR1_ACTIONS=false`，与冻结协议要求的
  `generation_mode=answer_only`、`allowed_actions=["answer"]` 不一致。
- 失败发生在三组 Judge 启动和 grouped labels 开启之前；
  `grouped_validation_labels_opened=false`、
  `final_dev1000_labels_opened=false`、0 evaluation、无 OOM。15 文件的
  完整失败树 SHA256 为
  `9947167a979294185f8900e4c953062825459c23f5094e113e382828263c49a9`，
  原 shadow 保留不覆盖。
- 经批准的 v5 recovery 只恢复冻结协议原本要求的服务生命周期：
  暂停 Native Reasoner/Retriever，启动独立的约束 answer-only 服务，
  要求 `/gateh0-health` 精确返回 `answer_only` 和仅 `answer`，对同一
  245 状态运行相同 prompt、model weights、生成预算和顺序；probe
  结束后停止该服务，再启动 Qwen3.5-2B Judge。final supervisor 在启动
  前同步加入同一服务生命周期，避免 dev1000 重复该已知错误。
- v5 不改模型、prompt、数据、顺序、预算、阈值、统计、seed、
  checkpoint 或 trajectory，不提前打开标签，也不重训。39 项相关测试
  PASS；旧 grouped/final supervisor 已逐字节备份，新 SHA256 分别为
  `959ba64aa6a661138fd850279efd2eaad749719c927e75b27261169571456319`
  与
  `9b34193dbe63b6a535a2cba9687105cb8ae17ddab5fb49a23fefb862fb626a68`。
- v5 已在新 shadow 完成 answer-only probes：health 精确 PASS、容器
  exit 0 且无 OOM；245/245 probes 全部为 `ANSWER`、parse rate 1.0、
  0 error、全部结构检查为 true。grouped/final labels 在审计时仍封存，
  随后自动进入 Structured Base Judge 标签盲评分。
- 小体积无标签证据已同步：
  `results/s2g-scaleup-aligned-eval-v1/grouped-v5-blind/`。
- amendment：
  `deploy/searchr1-v02/s2g-scaleup-answer-only-service-recovery-amendment-20260730.json`
- incident 与部署审计：
  - `results/s2g-scaleup-aligned-eval-v1/monitoring-incidents/20260730T075947+0800-answer-probe-native-service-protocol-failure.json`
  - `results/s2g-scaleup-aligned-eval-v1/answer-only-service-recovery-deployment-audit.json`

### 2026-07-30 final answer-only alias 前瞻兼容与隔离偏差披露

- v5 grouped probes 运行且 final 尚未创建 status/output 时，前瞻代码
  审查发现 final supervisor 虽已启动并 health-check 专用 answer-only
  服务，但 probe 命令仍传普通 Native `MODEL_ALIAS`。这会在 final
  dev1000 的盲 probe 阶段造成服务 alias 不匹配。修订只把该参数改为
  已冻结的 `ANSWER_ONLY_ALIAS`；旧/新 final supervisor SHA256 为
  `9b34193dbe63b6a535a2cba9687105cb8ae17ddab5fb49a23fefb862fb626a68`
  与
  `bfe1e4a99e474a7ace583a4929aba8e9dfe4be923d3efe9febe1ef609ffe0905`。
  原文件逐字节备份，33 项相关测试和远端静态 preflight PASS，当前
  grouped 未重启。
- amendment：
  `deploy/searchr1-v02/s2g-scaleup-final-answer-only-alias-compatibility-amendment-20260730.json`
- 同一次前瞻核对中，operator 误对封存的 final labels 文件执行了
  `sha256sum`。没有解析或显示任何 label 值、没有计算科学指标，唯一
  观察值是早已冻结在 supervisor 中的文件哈希；实验进程仍未挂载或
  打开 grouped/final labels，预测、checkpoint、阈值和配置均未改变。
  但文件字节读取本身违反严格的 no-read 程序边界，因此作为 isolation
  nonconformance 永久披露，不以重跑掩盖：
  `results/s2g-scaleup-aligned-eval-v1/monitoring-incidents/20260730T083100+0800-final-label-file-sha256-preblind-read.json`。

### 2026-07-30 10:05 Structured Base grouped 盲审计

- Structured Base 已完成相同冻结 grouped 状态上的 331/331 次评分；
  run_start/run_end 各 1，顺序完全一致、sequence 连续。
- 无标签审计 PASS：0 request error、0 generation error，320/331 可解析，
  parse rate 0.9668（门槛 0.95），所有 score 有限；11 个解析失败按冻结
  规则 fail closed 为 continue。
- 审计确认 0 gold field、labels 未挂载/未打开；grouped/final labels
  仍为 false。
- 容器 running、OOM=false、restart=0；链路自动进入旧版 S2G LoRA，
  没有重跑 Structured Base。

### 2026-07-30 11:00 旧 LoRA 审计失败与 fail-closed 恢复

- 旧 S2G LoRA 已完成 331/331 个冻结状态，0 request error、
  0 generation error，324/331 可解析，parse rate 0.9789；顺序、
  sequence、run_start/run_end、有限分数和无标签边界均正确。
- v1 审计如实 FAIL：7 条 `parsed=null` 记录虽然全部已由 runner
  输出 `decision=CONTINUE`，但其中 3 条 provenance-only 的首 token
  raw margin 大于等于 0。旧审计额外要求 raw margin 小于 0，且旧下游
  会直接消费该 raw margin，因而与预注册的“解析失败 fail closed 为
  CONTINUE”语义不一致。
- 失败时 grouped/final labels 均未打开，扩充版 LoRA 尚未启动；
  Judge 容器 `OOMKilled=false`。exit 137 来自 supervisor 在
  `finally` 中停止 sleep 容器，不是 OOM。失败输出、审计、日志和哈希
  已 append-only 保存。
- 显式兼容 amendment 将 policy score 语义澄清为：
  - 可解析记录继续使用原始有限 margin，逐值不变；
  - 显式 `parsed=null` 记录保留 raw margin 仅作 provenance，但
    effective score 固定为 `-1e30`，并要求 greedy decision 为
    `CONTINUE`；
  - 不含 `parsed` 字段的旧 binary 记录继续使用原 score。
- 该修订不改模型、prompt、数据、顺序、预算、阈值规则、统计、
  checkpoint 或任何已生成模型输出。冻结旧 run 的新审计已经 PASS：
  parse rate 仍为 0.9789，7 条解析失败全部 fail closed，其中 3 条正
  raw margin 被安全 mask。
- append-only recovery 复用 answer probes、Structured Base run 和
  旧 LoRA run，不重跑两组已完成推理；只重审并运行尚未开始的扩充版
  LoRA。启动快照为 expanded run 8 行、容器 running、OOM=false，
  grouped/final labels 仍为 false。
- final supervisor 已使用冻结 runner
  `run_searchr1_v02_smoke.scaleup-v2.py`（SHA256
  `0e04658edc905eae3fe5f0ce3e2ab3d9637222bfe0259895ef7348aaa3aaf4bb`）
  启动为 `waiting_for_gate_s4`。该等待状态不会运行 final 轨迹、打开
  final labels 或重选 checkpoint/阈值；Gate S4 完成后才自动衔接。
- amendment：
  `deploy/searchr1-v02/s2g-scaleup-failclosed-effective-score-recovery-amendment-20260730.json`
- 失败证据：
  `results/s2g-scaleup-aligned-eval-v1/grouped-v5-old-lora-audit-failure-20260730/`

### 2026-07-30 11:50 Gate S4 结果

- 三路 grouped 盲审计全部通过后才开启 grouped labels；final labels
  仍未打开。Gate S4 执行审计 PASS，阈值文件 SHA256 为
  `fcac089fb07a84342128769d245add2b8a2b58077645a13260fa3b9fe9d3b3f5`，
  决定文件 SHA256 为
  `2ad9bf9d4d3134bc72d843d1eb03a3f8e4f20e2f6b36014976c0844180721225`。
- grouped validation 共 245 个候选状态，其中 83 个安全 STOP 正例。
  扩充版 S2G LoRA 的冻结阈值为 `7.874999982333975`：
  STOP precision `0.9091`、recall `0.1205`、AP `0.5285`，
  触发 11 次提前停止，其中 10 次安全、1 次不安全。
- 在 100 个 grouped 问题上，扩充版 official EM 为 `0.53`，与
  Native Search-R1 的 `0.53` 相同；平均搜索次数从 `2.45` 降到
  `2.33`。本阶段没有 correct-to-wrong 或 wrong-to-correct 翻转。
- Structured Base 与旧 S2G LoRA 在预注册 precision≥0.90 且至少
  10 次 STOP 的阈值约束下均不可行，按协议退化为 never-stop；
  两者状态级 STOP AP 分别为 `0.4135` 和 `0.4585`。扩充版的
  grouped 结果支持“学到了部分安全提前停止信号”，但仍只是阈值选择集
  的初步证据，不能替代 final dev1000 的同题确认性结论。

### 2026-07-30 final 服务就绪失败与 append-only 恢复

- Gate S4 完成后，旧 final supervisor 对刚恢复的
  Reasoner/Retriever 做一次性 health 请求，遇到
  `ConnectionResetError` 并按协议停止。失败发生在 trajectory 启动前：
  final labels 未打开、未挂载，三个 final 容器均未创建，
  `final-dev1000/` 中 0 个输出文件。
- 失败 status、stderr、stdout 与旧 supervisor 已原样封存到
  `results/s2g-scaleup-aligned-eval-v1/final-service-readiness-failure-20260730/`。
  status SHA256 为
  `9831055c036b11961369209a3b954c901bc417b3f463c2efb4889945921e0fce`。
- 显式 recovery 仅将一次性检查改为两个服务连续两次 HTTP 200，并使用
  新目录 `final-dev1000-v2` 和新 status 文件；模型、prompt、数据与
  顺序、检索器、搜索预算、阈值、统计、checkpoint 均不变。
  amendment：
  `deploy/searchr1-v02/s2g-scaleup-final-service-readiness-recovery-amendment-20260730.json`。
- 恢复 supervisor SHA256 为
  `749b770f97ead9222a87b1fe8fcd8dccb7817474b5d33b8024ec4294b13225c4`；
  16 项定向测试与全仓 680 项测试 PASS，`git diff --check` PASS。
  恢复运行的服务就绪审计已以 2/2 连续成功通过，当前进入
  `native_trajectory_running`，容器 running、OOM=false，
  final labels 继续封存。

### 2026-07-30 final 输出命名空间前瞻桥接

- 恢复运行使用 canonical 目录 `final-dev1000-v2`，但已加载到进程中的
  supervisor 有两个 Judge 容器路径仍冻结为
  `/run/final-dev1000/states/...` 与 `/run/final-dev1000/blind/...`。
  若不处理，Native 轨迹与 answer-only probes 完成后会在 Judge 阶段
  因旧空命名空间而失败。
- 该问题在 Native 轨迹运行中、probes/Judge 尚未开始、final labels
  未打开/挂载时发现。旧 `final-dev1000` 只有空 `blind/evidence`
  目录，0 regular file；失败证据的 canonical 副本已另行冻结。
- append-only bridge 仅新增一个 states 相对软链接和三个 Judge 输出
  逐文件相对软链接，全部解析到 `final-dev1000-v2`。它不修改已有
  trajectory、不重启进程，也不改变模型、prompt、数据、顺序、检索器、
  预算、阈值、统计或 checkpoint。
- 机器审计 `final-output-namespace-bridge-audit.json` PASS；
  helper SHA256 为
  `7a048006dfe9468c7bd21ff12d926e1241dfc292d302f187c35bcfa963927c6d`。
  19 项相关测试和全仓 683 项测试 PASS，`git diff --check` PASS。
- amendment：
  `deploy/searchr1-v02/s2g-scaleup-final-output-namespace-bridge-amendment-20260730.json`。

### 2026-07-30 12:18–12:36 final Native 前缀审计

- 为在不读取 final labels、也不计算中间科学指标的前提下验证长时运行，
  新增标签盲前缀审计器。它只读取冻结 inputs 与轨迹文件的一致性字节
  快照，检查 `run_start`、错误数、问题 ID 唯一性、序号连续性、冻结
  输入 SHA、输入前缀顺序和隔离 guard；不读取、哈希或挂载 labels。
- 第一版监控审计在 12 条轨迹时 FAIL，但失败原因只是审计器把安全的
  负值 guard `gold_fields_received=0` 与 `labels_mounted=false`
  错当成 label payload。实验进程没有失败、没有重启或改变；失败审计
  与诊断作为监控器误报永久保留。
- 只修订监控审计器：精确允许上述两个安全负值，同时继续拒绝任何其他
  label/gold-like key。修订不改变模型、prompt、数据、输入顺序、
  检索器、预算、阈值、统计、checkpoint 或运行中的轨迹。
- 12:24 的 v2 审计在 14/1000 条轨迹时 PASS；12:36 的 v3 审计在
  23/1000 条时再次 PASS。v3 确认 1 个 `run_start`、0 个 `run_end`、
  0 trajectory error、23 个唯一问题 ID、连续序号、冻结输入前缀顺序
  完全一致，`gold_fields_received=0`、`labels_mounted=false`，
  无其他 label-like key，且 `scientific_metrics_computed=false`。
- v3 审计 SHA256 为
  `634473b34001c665c5c1e9a4a05be22d08b141340c8f63463ea9ed3d3b54c0bf`；
  审计器 SHA256 为
  `bfba48818de20eeeb22b337244e5da08b0aed798d92622c032eee7cf30cfb5ba`。
  同期容器 running、OOM=false、restart=0，trajectory stderr 为
  0 byte，final status 仍为 `native_trajectory_running`，final
  labels 仍未打开或挂载。
- v2 审计部署曾遇到一次 SSH 密码认证失败；连通性显式复核成功后只
  重做尚未完成的监控器部署，没有重跑或触碰实验。两次非科学事件分别
  记录在：
  `monitoring-incidents/20260730T122012+0800-final-prefix-audit-guard-field-false-positive.json`
  与
  `monitoring-incidents/20260730T122353+0800-prefix-audit-v2-deployment-auth-failure.json`。
- 本地全仓测试更新为 685 项 PASS，`git diff --check` PASS。

### 2026-07-30 13:19–13:23 final Native 增长窗口

- 在不读取、哈希或挂载 final labels 且不计算中间科学指标的条件下，
  以 45 秒间隔记录 6 个只读运行点。
- 轨迹由 47 增至 50，末端序号为 49，50 个序号连续且唯一，累计
  `trajectory_error=0`；容器在全部采样点均 running、OOM=false。
- final status 在窗口内保持 `final_labels_opened=false` 与
  `labels_mounted_during_blind_inference=false`。
- 快照 SHA256 为
  `622ba00815ed58c09bed493b2186342b3d6f1424e2e0636c63573e29dbb1029f`，
  并纳入 `monitoring-snapshots/artifact-hashes.sha256`。
- 该快照只证明输出持续增长和运行健康；输入前缀逐项一致性仍以
  23 条 v3 审计为权威，最终完成仍要求 1000/1000 终态结构审计。

### 2026-07-30 14:54 final Native 101 条前缀审计

- 轨迹越过 100 条里程碑后，使用同一标签盲审计器对 101 条前缀执行
  v4 审计；审计器 SHA256 仍为
  `bfba48818de20eeeb22b337244e5da08b0aed798d92622c032eee7cf30cfb5ba`。
- v4 PASS：冻结 inputs SHA256 精确匹配，101 个问题 ID 唯一，序号
  0–100 连续，输入前缀逐项一致，`trajectory_error=0`，
  `run_start=1`、`run_end=0`，符合仍在运行的状态。
- `gold_fields_received=0`、`labels_mounted=false`，无其他
  label-like key，且 `scientific_metrics_computed=false`。
- v4 快照 SHA256 为
  `057a775a7317822dff56cb8afcb880f9f8ecf90e3ff73afd5fbf5832cdf253bb`；
  轨迹字节快照 SHA256 为
  `b8d23659d452ed3eb58ba18f735c70939d4df965fdb9c1a3b23b25f54bf256da`。
- v1–v4 前缀审计已纳入
  `final-dev1000-prefix-audits/artifact-hashes.sha256`。该里程碑提高了
  运行中输入一致性证据强度，但最终完成仍要求 1000/1000 终态审计。

### 2026-07-30 15:31 前 200 题探索性中期检查

- 在 Native 轨迹完成 121/1000、final 标签仍未用于科学评测、尚未计算
  final 指标时，按用户要求冻结中期检查 amendment：
  `deploy/searchr1-v02/s2g-scaleup-final-interim-200-amendment-20260730.json`。
- 冻结 dev1000 的第 0–199 题为 **探索性前缀 200**，第 200–999 题为
  **确认性后缀 800**。原目标不缩减：三系统最终仍必须覆盖全部 1000
  题；完整 1000 题结果作为描述性同题总表，确认性主张与配对区间只由
  未触碰的后 800 题承担。
- 前 200 标签只有在 200 条 Native 轨迹通过冻结输入顺序、连续唯一
  序号、0 trajectory error、无 OOM、无 label-like payload 的强审计，
  且同前缀的 answer-only probes、Structured Base 与扩充版 S2G Judge
  输出均完成盲结构审计后，才允许由隔离评测器打开。
- 中期结果报告冻结阈值下的 STOP precision/recall、STOP AP、
  premature/unsafe STOP、official EM、平均搜索次数、节省搜索次数及
  correct↔wrong 翻转，只能解释为方向性初步结论。
- 不得根据前 200 题更改 checkpoint、模型、prompt、阈值、特征、预算、
  决定规则或指标。若发生任何科学配置修改，后 800 题不得继续沿用本
  amendment 的确认性资格，除非另行前瞻冻结新协议。

### 2026-07-30 15:58 中期运行分段协议

- 远端设备总 VRAM 为 32 GiB，Native 运行中约使用 23.6 GiB，余量约
  10.8 GiB。余量可容纳 Qwen3.5-2B Judge，但不足以并行加载第二份
  BF16 7B answer-only 服务，因此不能在 Native 不停机的情况下完成
  前 200 的全部 probe。
- 在 136/1000、前 200 边界尚未到达、final 科学标签未开启时冻结
  `deploy/searchr1-v02/s2g-scaleup-final-interim-segmentation-amendment-20260730.json`。
  方案是在至少 200 条完成记录后暂停并取得字节稳定快照，派生严格的
  0–199 前缀评测 run；probe 阶段后使用 runner 已有的显式 `--resume`
  继续完整 1000 题。
- 原始 JSONL 永久 append-only 保留。续跑必须写入 `run_resume`，其中
  `already_completed_count` 精确等于续跑前已落盘轨迹数，并跳过全部
  完成 ID。最终另行审计分段边界，再派生只含一个 `run_start`、1000
  个未改动 trajectory 对象和一个规范化 `run_end` 的评测 run。
- 如果暂停监控略晚于 200，额外完成题继续封存在后 800；探索性前缀
  仍严格只取 indices 0–199。没有落盘记录的在途题可在显式 resume 后
  执行一次；任何已经落盘的 `trajectory_error` 都会使流程 fail closed。
- 前缀冻结器与最终分段规范化器共 6 项定向测试 PASS，
  `git diff --check` PASS；两者代码 SHA256 已写入 amendment。
- 随后新增只接受无标签路径的 cut controller：它在达到边界后先
  `docker pause` 固定源字节，完成派生前缀与既有完整 run 结构审计后，
  才终止旧 Native 客户端；任何失败都写入状态且绝不自动续跑。三工具
  共 9 项定向测试 PASS。独立 code manifest：
  `deploy/searchr1-v02/s2g-scaleup-final-interim-segmentation-code-manifest-20260730.json`。
- 在 cut controller 尚未到达 200 时，又前瞻冻结纯盲下游 supervisor。
  它不接受 labels 路径、不调用 evaluator，只构建同前缀 state、运行
  专用 answer-only probes、Structured Base 与冻结扩充 LoRA，并要求
  全部结构审计 PASS。12 项相关定向测试和全仓 699 项测试 PASS；
  code manifest：
  `deploy/searchr1-v02/s2g-scaleup-final-interim200-blind-code-manifest-20260730.json`。
- 前缀 evaluator 只在全部盲审 PASS 后逐行读取前 200 个 label 对象，
  到边界立即停止，不哈希 full labels、不解析第 201 行；测试用例在
  第 201 行放置损坏 JSON 仍能 PASS。结果只标为 exploratory。
- 同时冻结 resume supervisor：它要求 raw run 与 cut 稳定快照完全
  一致，复用原 runner/full inputs/model/retriever/prompt/budget，以
  显式 `--resume` 追加边界；满 1000 后先做 raw 分段审计与规范化，
  再通过原完整结构审计。相关 18 项定向测试及全仓 705 项测试 PASS。
  code manifest：
  `deploy/searchr1-v02/s2g-scaleup-final-interim200-eval-resume-code-manifest-20260730.json`。
- 首次远端静态 preflight 因检查了 shadow 中的通用 runner 路径而退出；
  evaluator/resume 均未启动，Native/cut 守卫未受影响，0 label read。
  Docker bind mount 与活动 supervisor 命令证明权威 runner 实为 source
  repo 的 `run_searchr1_v02_smoke.scaleup-v2.py`，其哈希正是冻结的
  `0e0465…4bb`。失败事件永久保留，并以前瞻 path-only amendment
  修正 resume 参数；runner 字节和科学配置均未改变。
- 16:27 的显式恢复预检已对权威 runner、prefix evaluator、resume
  supervisor 和两份 amendment 副本逐项做 SHA256 与 Python 编译检查，
  全部 PASS；当时 cut controller 仍健康等待 200 题边界（156 条完成、
  indices 0–155 连续唯一、0 trajectory error），Native 容器
  running、OOM=false。此前一次短暂 SSH 密码拒绝已单独留证，随后仅用
  `true` 做认证诊断并恢复；没有启动 evaluator/resume，也没有读取或
  哈希标签。

### 2026-07-30 18:15 前 200 冻结与 SIGKILL 恢复

- 17:59 达到 200 条后，cut controller 成功冻结严格 indices 0–199：
  `prefix-freeze-audit` 与 `prefix-structure-audit` 均 PASS，200 个 ID
  连续唯一、0 trajectory error、0 label read/hash、0 科学指标。
- controller 解除暂停后发送的 `docker kill --signal=SIGTERM` 返回 0，
  但容器内 Python PID 1 未退出；旧 supervisor 持续阻塞于
  `docker wait`，120 秒后 controller 按预注册门限 fail closed。原失败
  status、容器 inspect 和 kill 日志全部保留，没有启动盲 supervisor。
- 在标签仍封存时冻结显式基础设施 amendment
  `s2g-scaleup-interim-cut-sigkill-recovery-amendment-20260730.json`。
  恢复脚本先再次暂停 Native，记录实际 append-only 进度与 SHA256，再
  发送 SIGKILL；只有暂停至终止期间 raw JSONL 字节完全一致、容器退出、
  旧 supervisor 写入 sealed failure 时才生成新的完成状态。3 项定向
  测试及全仓 708 项测试 PASS，`git diff --check` PASS。
- 18:14 恢复 PASS。严格前缀哈希保持不变；raw run 在受控终止时为
  205 条、0 error，暂停/终止 SHA256 同为
  `590cd71371b792e2811b03419b0589b5cafea6ee3799d3d133b0b79f866347bc`。
  多出的 5 条没有进入前 200 探索集，仍属于后 800 确认性 suffix，并
  将成为显式 resume 的 `already_completed_count=205`。
- 18:15 以新的恢复 status 启动一次纯盲 supervisor；其 cut status
  SHA256 为 `b05b7474…617dcd`。已冻结 529 个 answer-probe 状态并进入
  `answer_probes_running`，标签仍未打开，尚未计算科学指标。

### 2026-08-03 Gate S5 终态

- Native 经过唯一的多 resume recovery 完成 1,000 题。resume1 的
  Retriever OOMKilled/exit137 与 index626 连接错误原样保留；resume2
  只在用户授权后增加 16 GiB 非持久 swap 并从唯一 626 边界继续，模型、
  Retriever 配置、prompt、预算、输入顺序和统计均未改变。
- 派生 resolved-infrastructure view 只排除已被同题成功轨迹解决的 1 条
  基础设施 error；规范化结构审计为 1,000 条、0 error、唯一且顺序
  连续、run_start/run_end 各 1。answer probes 与三路 Judge 盲审全部
  PASS 后，final labels 才在 2026-08-03T06:00:37Z 打开。
- indices 200–999 的确认性结果：扩充版相对 Native 的 EM 差
  −0.00625，95% CI [−0.01250, 0]；搜索次数差 −0.09625，95% CI
  [−0.12000, −0.07375]。两道冻结硬门均 PASS，科学 outcome 为
  `SYSTEM_GATE_PASS`。
- 全 0–999 的描述性结果：Native/扩充版 EM 为 0.449/0.443，平均搜索
  为 2.610/2.521。不得用该全量表替代 suffix-800 确认性区间。
- final evaluator 的 normalized-structure 文件名不兼容与随后 launcher
  module import 失败均在 label 开启前 fail closed；旧状态永久保留，
  第二次经用户授权的 module invocation 唯一成功。
- 冻结科学产物：`final-dev1000-v2/evaluation.json`、
  `evaluation-confirmatory-suffix800.json`、`decision.json`、
  `material-passport.md` 与 `artifact-hashes.json`。完整 incident 与主张
  边界 companion 为
  `results/s2g-scaleup-aligned-eval-v1/material-passport-incidents-and-claims.md`。
