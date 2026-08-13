# Gate H0：Search-R1 停止条件上界审计实验协议

## Material Passport

- **Artifact ID**：`searchr1-v02-gate-h0-stop-headroom-protocol`
- **Artifact type**：探索性实验协议 / Critic 前置资源门禁
- **Status**：设计已冻结；v2 answer-only 执行已于 2026-07-26 完成并通过
- **Frozen date**：2026-07-26
- **Primary dataset**：HotpotQA exploratory pilot 200
- **Upstream system**：Search-R1 v0.2，Gate A
  `CONFIGURATION-VERIFIED/COMPATIBLE`
- **Primary run**：
  `results/searchr1-v02-reproduction/gatec-precheck-200/remote/forced-reject-20260726/precheck-forced.jsonl`
- **Native replay evidence**：
  `results/searchr1-v02-reproduction/gatec-precheck-200/legacy-pilot/report/pilot-question-comparison.jsonl`
- **Evidence role**：仅用于可研究性、上界和资源分配；不是 Gate C 或
  final-test 确认性证据
- **Label status**：这 200 题的 answer labels 已在既有 exploratory
  pilot 中查看；本协议只能冻结在 probe 输出产生前，不能追溯声称为
  label-unseen 预注册
- **Downstream gate**：Gate H0 决定是否以及以何种形式继续停止器研究；
  不预先绑定 SIM-RAG

## 1. 目的

Gate H0 不训练 Critic，也不比较具体停止算法。它先回答两个更基础的
问题：

1. Search-R1 是否会在已有状态足以支持正确作答时继续执行搜索？
2. 在冻结 Search-R1、Retriever、语料、最大预算和 answerer 后，只改变
   停止时机，最多能提高多少答案质量或节省多少搜索？

本实验得到的是**冻结可达状态与固定 finalizer 下的经验条件上界**，
不是所有查询、文档、模型和提示词上的全局数学上界。

## 2. 假设与可证伪结论

### H0-A：质量型过度搜索

存在原生下一动作为 `SEARCH` 的状态，固定 stop-now finalizer 能在该
状态生成 official-EM 正确答案，而 Native 最终答案错误。

这类题构成 `harmful over-search rescue`，决定仅改变停止时机是否具有
质量增益上界。

### H0-B：成本型过度搜索

存在原生下一动作为 `SEARCH` 的状态，固定 stop-now finalizer 已能生成
正确答案，且 Native 最终也正确，但 Native 使用了更多搜索。

这类题构成 `cost-only over-search`，决定在不降低 Native EM 时最多能
节省多少检索。

### 失败条件

若质量 headroom 和 Native-preserving 成本余量都很小，则该冻结
Search-R1 实例上缺少足够的停止可辨识空间，不应继续投入大规模 Critic
训练。

## 3. 状态集合与归因边界

### 3.1 主分析：Native 可达前缀

既有 200 题最长轨迹共有 598 次成功检索，但其中一部分发生在原生候选
被实验控制器强制 Reject 之后。根据冻结的 `native_first_answer` 离线
回放：

- 问题数：200；
- Native 成功检索总数：523；
- Native 可达状态数：`200 + 523 = 723`；
- 原生下一动作为 `SEARCH` 的状态数：523。

每题的 Native 可达状态定义为：

```text
s_i,0 = 初始问题状态
s_i,k = 第 k 次 Native 成功检索后的 rolling state，k = 1..K_i
```

主分析只使用这 723 个状态。`Stop-now answerable SEARCH rate` 的分母
是 523 个原生 `SEARCH` 动作状态，而不是全部 723 个状态。

### 3.2 补充分析：forced-continue 反事实后缀

forced-reject 最长轨迹比 Native 前缀多 75 次成功检索，因此另有 75 个
原生答案被拒绝后才出现的 post-search 状态。它们不是 Native 实际会
到达的状态，必须标记为：

```text
counterfactual_forced_continue = true
```

这些状态只能用于补充的候选可达性或未来 Bidirectional Action Oracle，
不能用于声称“Native Search-R1 仍然继续搜索”。

### 3.3 总运行规模

- 主分析：723 个 probes；
- 可选反事实补充：75 个 probes；
- 若一次性全部运行：798 个 probes。

报告、表格和图必须把 723 与 75 分栏，不能合并成一个 Native over-search
比例。

## 4. Stop-now answer probe

### 4.1 设计原则

probe 不是重新构造一个 `question + cumulative passages` 的新 QA prompt。
每个 probe 必须从 Search-R1 在该状态实际看到的 rolling token state
继续，以保留：

- 原始问题和 Qwen chat framing；
- 先前 `<think>`、`<search>` 和 `<information>` 的顺序；
- 每轮 observation 的 Top-3 内容与截断；
- 4096-token rolling-window 语义；
- 反事实补充状态中的 Reject 反馈。

probe 不包含 gold answer、supporting-fact 标签、状态正确性、`complete`
条件或任何“证据已经充分”的暗示。

### 4.2 冻结 finalizer 文本

主 finalizer 的英文文本精确冻结为：

```text
Search is no longer available. Using the question and all information currently available in the conversation, conduct reasoning inside <think> and </think>, then provide one concise final answer inside <answer> and </answer>, without detailed illustrations.
```

不允许在看到 probe EM、Oracle 或错误案例后改写该文本。

该措辞有意不使用以下表达：

- `You have enough information`；
- `The answer is in the evidence`；
- `Use only the retrieved documents`；
- `Guess if necessary`；
- 任意 gold answer、支持句或证据完整性标签。

### 4.3 完整 prompt 结构

`S_i,k` 表示原生 rolling state 的 token 序列。probe 在关闭当前 assistant
turn 后增加一个新的固定 user turn：

```text
<原生 Search-R1 rolling state S_i,k><|im_end|>
<|im_start|>user
Search is no longer available. Using the question and all information currently available in the conversation, conduct reasoning inside <think> and </think>, then provide one concise final answer inside <answer> and </answer>, without detailed illustrations.<|im_end|>
<|im_start|>assistant
```

实现必须用冻结 checkpoint 的 tokenizer 生成特殊 token，不得把
`<|im_start|>` 或 `<|im_end|>` 当普通文本近似处理。最终输入为：

```text
probe_input_ids = (rolling_state_ids + finalizer_suffix_ids)[-4096:]
```

必须保存：

- `state_id`、`question_id`、`state_index`；
- `native_prefix` 或 `counterfactual_forced_continue`；
- 原生下一动作；
- `rolling_state_token_count`；
- `finalizer_suffix_token_count`；
- `probe_prompt_token_count`；
- `rolling_state_sha256`；
- `finalizer_text_sha256`；
- `finalizer_suffix_ids_sha256`；
- `probe_input_ids_sha256`；
- 截断后问题 token 序列是否仍完整保留。

### 4.4 为什么不复用旧 counterfactual prompt

`kstar/searchr1_counterfactual_audit.py` 中的
`canonical_b_v1` 会人为加入：

```text
<think>I need external evidence to answer the question.</think>
<search>{question}</search>
```

它是旧证据敏感性实验的 Search/Answer decision probe，会主动诱导搜索，
也不保留真实多轮 rolling state，因此禁止用于 Gate H0。

`direct_information_a0` 同样禁止作为主 prompt，因为它重新拼接 question
和 evidence，丢失真实查询、推理、顺序、Reject 反馈和 token 截断语义。

## 5. 模型与解码冻结

probe 使用与 Gate A 相同的 Qwen2.5-7B Search-R1 checkpoint 和 tokenizer：

```text
temperature = 0
top_p = 1
repetition_penalty = 1
max_new_tokens = 500
max_prompt_tokens = 4096
do_sample = false
constrained_decoding = false
retry_on_search_or_invalid = false
```

只允许基础设施级重试完整请求；不得因为模型输出 `SEARCH`、非法 action
或错误答案而追加第二次作答提示。

## 6. 解析和结果语义

每个 probe 只运行一次，并使用 Search-R1 v0.2 的 official postprocess 与
action parser：

- 首个合法 `<answer>...</answer>`：记录候选答案；
- 输出 `<search>...</search>`：记为 `probe_search`，策略结果为 abstain；
- 无合法 action：记为 `probe_invalid`，策略结果为 abstain；
- completion、timeout 或服务异常：记为 infrastructure error，Gate H0
  结构审计失败，不能当成模型 abstain。

答案使用与 Gate A/Gate C-0 相同的 official normalization 和 EM；Token F1
只作补充，不参与 Oracle 选择。

## 7. 三层上界与主要指标

### 7.1 Evidence Oracle

仅在能够从授权 HotpotQA source materialize supporting facts 时计算。
当前 pilot label materializer 只读取 `data_source` 和 `reward_model`，
不能假定已经拥有完整 supporting-fact 标签。

若支持事实不可用：

- `Evidence-ready SEARCH rate` 标记为 `UNAVAILABLE`；
- 不得用 answer string containment 或 LLM judge 冒充 Evidence Oracle；
- Quality/Cost Oracle 仍可正常执行。

### 7.2 Quality Stop Oracle

令 \(c_{i,k}=1\) 表示问题 \(i\) 在状态 \(k\) 的 probe answer 与 gold
official-EM 匹配：

```text
Quality Stop Oracle(i) = 1, if max_k c_i,k = 1
Quality headroom = Oracle EM - Native EM
```

若存在多个正确状态，Oracle 选择搜索次数最少的最早状态。

### 7.3 Native-preserving Cost Oracle

只对 Native 最终正确的问题选择最早正确 probe；Native 最终错误的问题
保持 Native 行为不变。报告：

- 保持 Native EM 时的平均搜索数；
- 每题与总体可避免搜索数；
- `avoidable_search_fraction`；
- 问题级 paired bootstrap 描述性区间。

### 7.4 原生 SEARCH 四格分解

只在 523 个原生 `SEARCH` 动作状态上统计：

| Stop-now probe | Native 最终结果 | 解释 |
|---|---|---|
| 正确 | 正确 | cost-only over-search |
| 正确 | 错误 | harmful over-search rescue |
| 错误/abstain | 正确 | necessary or beneficial search |
| 错误/abstain | 错误 | unresolved/futile search |

同一问题可能有多个 `SEARCH` 状态。动作级比例和问题级“是否至少出现一次”
必须分开报告，不能把状态当成独立问题样本计算显著性。

### 7.5 固定预算曲线

报告 `k=0..4` 的 forced-finalization EM、Token F1、平均搜索数和有效状态
覆盖。不存在第 \(k\) 个 Native 状态的问题按预先冻结的 last-reachable
规则取最后一个 Native 状态，不允许因答案正确性选择替代状态。

## 8. 探索性资源分支

- `Quality headroom >= 5pp`：停止控制具有强质量空间；
- `2pp <= Quality headroom < 5pp` 且可避免搜索至少 20%：
  进入质量—成本 Pareto 主线；
- `Quality headroom < 2pp` 但成本余量明显：
  转为效率型停止研究；
- 质量与成本余量都很小：
  不投入大规模 Critic 训练。

这些阈值只用于 200 题探索性资源分配，不是统计显著性、风险认证或
论文结论门槛。

## 9. 运行顺序与标签隔离

1. 从既有 forced-reject 文件和 `native_first_answer` 回放结果生成状态
   manifest；
2. 自动断言 200 个唯一问题、523 次 Native 搜索、723 个 Native 状态、
   75 个反事实 post-search 状态；
3. 冻结 prompt 文本、tokenized suffix hash、代码 revision、容器和输出
   schema；
4. 先运行 20 个无标签 smoke states，仅检查 token reconstruction、解析、
   latency 和基础设施；
5. smoke 不读取答案、不计算 EM，也不得依据输出内容改 prompt；
6. 运行 723 个主 probes；75 个反事实 probes 可在同一批次运行，但输出
   role 必须分离；
7. 完成无标签结构审计后，在隔离评测步骤中连接已属于 exploratory pilot
   的 answer labels；
8. 生成中文报告、逐题结果、结构审计、决定文件和 SHA256 清单；
9. final-test labels 全程保持不可用。

## 10. 预计耗时

已有 330 个原生 answer actions 的平均延迟为 8.12 秒：

- 723 个主 probes 的顺序推理点估计：约 1.63 小时；
- 75 个反事实补充 probes：约 0.17 小时；
- 全部 798 个 probes：约 1.80 小时；
- 考虑长 context、服务启动和波动：主分析预算 1.8–2.2 小时，全部状态
  预算 2–2.5 小时；
- 状态准备、smoke、结构审计、隔离评测与报告：约 0.5–1 小时；
- runner 实现与测试：约 1.5–2.5 小时。

从当前状态开始，包含实现、测试、运行和报告，保守总耗时仍为 4–6 小时。
20-state smoke 后只允许用实际吞吐更新 ETA，不允许修改模型、prompt、
状态集合、解析或评价口径。

## 11. 必需产物

```text
deploy/searchr1-v02/gateh0-stop-headroom-protocol-20260726.json
results/searchr1-v02-reproduction/gateh0/state-manifest.jsonl
results/searchr1-v02-reproduction/gateh0/run.jsonl
results/searchr1-v02-reproduction/gateh0/structure-audit.json
results/searchr1-v02-reproduction/gateh0/quality-cost-report.json
results/searchr1-v02-reproduction/gateh0/quality-cost-report.md
results/searchr1-v02-reproduction/gateh0/gateh0-decision.json
results/searchr1-v02-reproduction/gateh0/artifact-hashes.sha256
```

正式 protocol JSON 必须逐字包含本文件冻结的 finalizer、状态角色、解码、
指标和失败规则；不得只链接本文档而省略关键字符串。

## 12. 审查结论

Gate H0 的主问题应表述为：

> 在 Search-R1 的原生可达状态上，固定 finalizer 能否在原生策略选择继续
> 搜索时恢复正确答案，以及这种机会对应多少质量和成本上界？

不能表述为：

> 所有 798 个状态都证明 Native Search-R1 在过度搜索。

前者由 723 个 Native 状态和523个原生 `SEARCH` 动作支持；后者错误地把
75 个 forced-continue 反事实状态归因给 Native。
