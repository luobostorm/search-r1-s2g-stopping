# 多轮 RAG 何时停止：论文核心立意

## 1. 中心问题

本文研究的核心问题不是某个 Critic 在特定迁移配置下是否失败，而是：

> 多轮 RAG 系统应在什么时候停止检索；在优化停止器之前，如何证明强搜索策略确实会在已有状态足以正确作答时继续搜索；这种停止错误在冻结模型、Retriever 和预算后的条件上界有多大；以及如何证明一个停止策略在答案风险、覆盖率和搜索成本之间实现了有效权衡？

Search-R1 是使用官方 checkpoint、Retriever、语料、提示词和预算，并在预注册分层样本上达到 `CONFIGURATION-VERIFIED/COMPATIBLE` 的强多轮搜索基线；SIM-RAG 是外部停止控制器的代表。二者是研究这一问题的实验载体，不是论文立意本身。

截至 2026-07-25，answer-event 接口的 Gate B 已通过：官方 SIM-RAG Critic 在 22 个冻结 Search-R1 候选状态上产生非退化 margin，margin=0 的离线 earliest-stop 与 20 题在线复跑 20/20 严格一致。该结果只证明接口与回放方法有效，不证明停止策略降低了答案风险；后者必须由独立 calibration 和 final test 回答。

截至 2026-07-26，200 题 Gate C-0 标签盲工程审计也已通过：
201/201 个状态成功打分，186 个唯一 margin，margin=0 会否决 54 个首
候选并改变 24 个最终答案。随后在结果角色、固定策略和不调阈值均预先
写明的探索性旁路中，比较了 Native、Fixed-min-2、released SIM-RAG
margin=0 与 official-EM Trajectory Oracle。该旁路已查看这 200 题的
calibration labels，但未读取 1,000 题 final-test labels；它只提供
方向与失败分解，不是 Gate C 确认性证据。

| 200题探索策略 | Official EM | Token F1 | 主动 coverage | 主动经验错误率 | 平均搜索 |
|---|---:|---:|---:|---:|---:|
| Native Search-R1 | 46.5% | 58.54% | 83.0% | 49.40% | 2.615 |
| Fixed-min-2 | 46.5% | 58.54% | 83.0% | 49.40% | 2.685 |
| Released SIM-RAG，margin=0 | 42.5% | 54.42% | 56.5% | 38.94% | 2.710 |
| Trajectory Oracle（official-EM 后验） | 47.0% | 47.00%* | 42.0% | 0.00% | 2.785 |

\* Oracle 在没有 official-EM 正确候选时 abstain，且优化目标是 EM，
因此该 F1 不是 token-F1 上界。

SIM-RAG 相对两个可部署基线均为 0 次救援、8 次破坏，配对 EM 差
-4.0pp，paired bootstrap 95% 区间为 [-7.0pp, -1.5pp]。Trajectory
Oracle 仅在 94/200 题中找到 official-EM 正确候选；Native 已捕获其中
93 题，released SIM-RAG 只捕获 85 题。这组数据暂不支持“released
Critic 的停止决策更好”，但直接支持本文必须把候选生成上限、选择性
风险、coverage、端到端质量和搜索成本分开报告的核心立意。

## 2. 文献定位

顶会文献中的“自适应检索”至少包含四种不同决策接口：

1. 查询执行前的 no/single/multi-step 路由；
2. 证据收集过程中的 Continue/Terminate；
3. 生成句段中的检索触发；
4. 已有候选答案上的 Accept/Reject。

GMR、IRCoT、FLARE、Self-RAG、Adaptive-RAG、IM-RAG、EfficientRAG、
Search-R1、SIM-RAG、ReSearch 和 ACL 2025 统一评测的实验设置详见
[`多轮RAG停止策略-顶会实验设置文献调研.md`](多轮RAG停止策略-顶会实验设置文献调研.md)。

与本文最直接相同的是 SIM-RAG 的答案级停止，但原生 SIM-RAG 每轮都
生成 answer+rationale；当前 Search-R1 answer-event veto 只在 Search-R1
自行生成 `<answer>` 时才有候选。因此，当前低 Oracle headroom 首先是
接口可研究性问题，不能未经验证就归结为“停止研究没有意义”。

## 3. 中心论点

> 多轮 RAG 停止不是独立状态上的二分类问题，而是一个由证据可达性、答案可恢复性、候选演化、首次越界、搜索预算和失败回退共同决定的序列选择问题。停止器只有在冻结的决策接口上存在足够 Oracle headroom 时才可被识别和优化；随后才应以问题级风险、覆盖率、端到端质量和成本联合评价。

第一阶段的可研究性审计不预设 Critic 形式，而把停止机会定义在
Search-R1 每个可达检索状态：初始状态及每次成功检索后的状态。对每个
状态运行一次离线、非干预式 stop-now answer probe，模拟“若此时停止并
由冻结的 7B Reasoner 作答，结果是什么”。probe 只读取当时已有的
question/context，不把输出反馈给 Search-R1，也不改变后续查询轨迹。
该诊断允许识别 Search-R1 提议继续 `SEARCH` 时已经能够正确作答的状态；
它不是未来在线系统必须承担的逐轮 7B 调用，也不等价于开放环境中的全局
数学上界。

## 4. 研究问题

1. 在官方配置且通过预注册一致性审计的 Search-R1 上，有多少原生 `SEARCH` 发生在证据已经充分或冻结 Reasoner 已能正确作答之后？
2. 在冻结 Search-R1、Retriever、语料和预算后，只改变停止时机的决策条件 Oracle 最多能提高多少 EM、避免多少搜索？
3. 在确认存在足够质量或成本余量后，外部停止器能否比 Native Search-R1、各固定预算和 matched-cost 策略做出更好的 Stop/Continue 决策？
4. 状态级 Critic 判别能力能否转化为问题级 earliest-stop 风险改善？
5. 停止性能的上限主要由证据缺失、答案生成、Critic 判断、首次越界规则还是预算耗尽后的系统行为决定？

## 5. 论证链

### 论点一：停止研究必须建立在可信的上游基线上

- 先在官方 checkpoint、E5、Wiki-2018、Top-3 和官方预算下验证 Search-R1 的实现与性能。
- 实现正确性由无标签链路、确定性复跑、协议断言和资源指纹保证；性能一致性由预注册的 NQ/HotpotQA 分层样本审计。
- 1,200 题审计中 NQ/HotpotQA official EM 为 45.67%/46.00%，论文参考值均位于 source-specific Wilson 95% 区间且点差小于 5pp；因此基线达到“配置已核验且结果相容”，但没有达到更强的 Wilson 90% 区间等价条件，正文不得称为严格复现。
- 完整 11,015 题官方测试集只属于可选附录复现，不作为进入停止实验的必要条件。
- 基线验证是内部有效性门槛，不是论文的主要算法贡献；计算预算优先投入候选上界和确认性停止策略比较。

### 论点二：状态判别与策略停止是不同的评价对象

- AUC/AP 衡量状态排序，不包含同一轨迹中的时间顺序。
- Earliest-stop 只由第一个越过阈值的状态决定。
- 较早的错误高分可以截断较晚的正确答案，因此高 AUC 不自动推出低 accepted risk。
- Gate B 的 20/20 policy consistency 只验证“同一冻结分数是否复现同一在线策略”，不使用 gold，也不能替代风险校准。
- 200 题探索中，margin=0 的主动经验错误率比 Native 低约 10.46pp，
  但 coverage 同时低 26.5pp，端到端 EM 低 4.0pp且搜索更多；这正是
  “选择性风险改善”不能自动写成“停止决策更好”的实例。

### 论点三：Critic 迁移和 Critic 方法能力必须分开

- Released SIM-RAG Critic 检验即插即用迁移。
- Search-R1-adapted SIM-RAG Critic 检验在目标轨迹上适配后的方法能力。
- 二者的差异用于识别 Reasoner、Retriever 和轨迹格式错配，而不是把所有失败归因于停止思想本身。

### 论点四：停止器必须与候选上限和失败回退一起解释

- 没有正确候选时，任何停止器都不可能返回正确答案。
- 未越过阈值后的 forced finish 不属于 Critic 主动接受。
- 因而必须同时报告 Oracle coverage、rescue rate、voluntary risk、coverage、forced finish、abstain 和成本。
- 200 题 official-EM Oracle coverage 只有 47.0%，106 题整条轨迹无
  正确候选；Native 对可达正确候选的捕获率已达 93/94（98.94%）。
  因而这批数据中的可改善空间主要不在“多搜一次”，而在生成新的有效
  候选和避免 Critic 破坏本来正确的早停。

### 论点五：先验证决策条件上界，再训练停止器

- Adaptive-RAG 使用 Oracle classifier 验证策略路由余量，SIM-RAG 使用
  gold Oracle Critic 扫描轮数；这两类实验都先展示控制问题的上限。
- 当前 answer-event Oracle 相对 Native 只有 +0.5pp，训练 adapted Critic
  无法突破候选集合本身。
- 因此新增 Gate H0：在初始状态和每次成功检索后的冻结状态上运行不反馈
  给策略的 stop-now answer probe，直接检验“Search-R1 已可正确作答却
  继续搜索”的发生率，并比较 Native、固定预算曲线和停止 Oracle。
- Gate H0 只决定是否值得继续投入 Critic 训练，不把同一 200 题 pilot
  变成确认性结果。

## 6. Gate H0：停止决策条件上界门禁

执行顺序：

1. 复用 200 题 exploratory pilot 的冻结最长轨迹，不重新运行 Search-R1
   或 Retriever；主分析提取 200 个初始状态和 523 个 Native 成功检索后
   状态，共 723 个原生可达状态；另将 forced-reject 后新增的 75 个
   post-search 状态标记为反事实补充；
2. 在 labels 不可见时冻结 stop-now probe prompt、chat template、确定性
   解码、最大 token、答案解析、状态去重和成本口径；
3. 标签盲运行全部 probe；输出不反馈给 Search-R1，不改变原生查询或后续
   状态；
4. 完成 723/723 个主状态以及可选 75/75 个反事实状态的覆盖、解析率、
   重复状态、token、延迟和哈希审计后，才在隔离评测步骤中连接已属于
   exploratory pilot 的 labels；
5. 对每个原生 `SEARCH` 前状态，将 stop-now probe 与 Native 最终结果
   组成四类：当前对/最终对、当前对/最终错、当前错/最终对、当前错/最终错；
6. 计算下列相互独立的上界和诊断指标。

主要指标：

- **Evidence-ready SEARCH rate**：原生搜索动作中，执行前证据已充分的比例；
- **Stop-now answerable SEARCH rate**：原生搜索动作中，执行前 probe 已
  official-EM 正确的比例；
- **Quality Stop Oracle**：每题在全部可达状态中后验选择最早正确答案；
- **Native-preserving Cost Oracle**：只对 Native 原本正确的题提前到最早
  正确状态，保持 Native EM 不变时估计最大可节省搜索；
- `Quality headroom = Stop Oracle EM - Native EM`；
- harmful over-search rescue：早期 probe 正确但 Native 最终错误；
- cost-only over-search：早期 probe 与 Native 最终都正确，但原生多搜；
- 每题可避免搜索数、总体可避免搜索比例和 `k=0..4` fixed-budget
  EM—成本曲线；
- evidence-ready/answer-wrong：证据充分但冻结 Reasoner 仍答错的生成差距。

这里的 Oracle 是“冻结可达状态和冻结 answerer 条件下的经验上界”，
不是所有可能查询、文档或模型上的全局理论上界。Native 归因只使用
723 个原生可达状态；75 个 forced-continue 状态只作反事实补充。完整
双向 Action Oracle 留到 Gate H0 通过后再决定，避免把实验控制器制造的
后缀误写成 Native 行为。

探索性资源决策分支：

- `Quality headroom >= 5pp`：停止控制具有强质量改进空间；
- `2pp <= Quality headroom < 5pp` 且可避免搜索比例不低于 20%：
  主问题转为质量—成本 Pareto 改进；
- `Quality headroom < 2pp` 但成本余量明显：转为效率型停止研究；
- 质量和成本余量都很小：不训练大规模 Critic，停止问题在该冻结基线上
  缺少足够可辨识空间。

这些阈值只用于 200 题探索性资源分配，不是统计显著性、风险认证或最终
论文通过条件。Gate H0 通过后再冻结具体停止器；不预先绑定 SIM-RAG。

Gate H0 已于 2026-07-26 完成：

- 20 状态 smoke 与 798 状态正式 answer-only probe 均为 100% 解析、
  0 error；723 个 Native 状态和 75 个反事实状态全部完成；
- 正式推理 7,380.05 秒，平均 9.248 秒/状态；完整结构审计通过前
  labels 不可见，之后才隔离评估；
- 523 个原生 `SEARCH` 状态中，143 个（27.3%）在搜索前已能正确
  stop-now；涉及 70/200 题（35.0%）；
- 60/200 题存在 cost-only over-search；10/200 题存在早期正确而
  Native 最终错误的 harmful over-search；
- Native EM 46.5%，Quality Stop Oracle EM 51.5%，精确质量增量为
  10/200=5.0pp；保持 Native EM 的 Cost Oracle 可避免
  131/523（25.05%）次搜索；
- `k=0..4` 的固定预算 answer-only EM 为 18.5%、34.0%、43.0%、
  46.0%、47.0%，显示统一预算不能直接实现问题级 Oracle；
- 75 个 forced-continue 状态的 52.0% EM 只说明反事实答案可达性，
  不归因为 Native 过度搜索；
- supporting-fact labels 未 materialize，所以 Evidence-ready SEARCH
  rate 保持 `UNAVAILABLE`，不把“答对”偷换为“证据充分”。

精确题数满足预冻结的 `>=5pp` 强质量分支。原始机器报告因
IEEE-754 边界表示保留 `QUALITY_COST_PARETO` 枚举；透明勘误把协议
解释记为 `STRONG_QUALITY_HEADROOM`，没有修改任何指标或阈值。

这改变了下一步资源判断，但不把 pilot 结果升级为论文结论：研究问题
具有足够经验上界，值得训练或设计状态自适应停止器；能否接近该上界仍
必须在新的 calibration/final-test 划分上与 Native、固定预算和
matched-cost 对照验证。

2026-07-27 的两个 Judge 训练实验进一步收窄了方法论点。Binary LoRA
提高状态排序 AP，却因过度 STOP 使 EM 下降；S2G 风格结构化监督把
错误 STOP 从 99 降到 24、premature STOP 从 18 降到 4，并将 EM 从
31.5% 恢复到 35.5%，但相对 Structured Base 的 +1.5pp 差异不确定，
且平均多搜索 0.395 次。由此，论文不能把“结构化 Judge 已经优于原生”
作为现成结论；更稳健的立意是：

1. 强 Search-R1 上确有可测停止上界；
2. 状态判别排序、停止安全性、端到端 EM 和搜索/推理成本可能相互冲突；
3. 结构化缺口监督是降低激进过停的候选机制，但必须通过独立 calibration
   和成本匹配对照才能转化为系统主张。

截至 2026-07-30，扩充版结构化 Judge 已完成独立数据与训练闭环：
HotpotQA train 按问题隔离为 700 train、100 grouped validation 和
200 reserve，Gate S3 最终使用 900 个训练问题、3,009 个 clean states。
Qwen3.5-2B S2G LoRA 完成 3 epochs / 12,654 steps，并只按自然分布
grouped validation 的冻结规则选择 epoch 1。Gate S4 在三路 Judge
标签盲审计通过后才开启 grouped labels；扩充版在冻结阈值下得到
STOP precision/recall/AP `0.9091/0.1205/0.5285`，100 题 official EM
与 Native 均为 `0.53`，平均搜索次数从 `2.45` 降到 `2.33`。

这组 grouped 结果只用于冻结 final 阈值。2026-08-03 完成的 Gate S5
严格使用未查看的 indices 200–999 作为确认性后缀：扩充版相对 Native
的 Official EM 差为 −0.00625，95% CI [−0.01250, 0]，满足冻结的
−0.02 非劣门；平均搜索差为 −0.09625，95% CI
[−0.12000, −0.07375]，满足搜索减少门。扩充版相对 Structured Base
的 STOP AP 差为 +0.03983，95% CI [+0.00571, +0.07228]，但固定阈值
STOP precision 只有 0.6216。因此论文主张应限定为“结构化 Judge 在
独立同题后缀上把状态排序信号转化为质量非劣界内的搜索减少”，不得写成
答案质量优于 Native，也不得宣称 STOP 风险已获认证。完整 0–999 结果
只作描述性总表。

具体状态角色、完整 finalizer、token 拼接、解析、标签隔离和失败语义见
[`Gate-H0停止条件上界审计实验协议.md`](Gate-H0停止条件上界审计实验协议.md)。

## 7. 主要贡献边界

本文可以主张：

1. 在官方配置、实现审计通过且结果与论文报告统计相容的强搜索基线上，先验证停止接口的候选可达性条件；
2. 受控比较 Native、固定预算、matched-cost、released SIM-RAG、
   结构化 Learned Judge 和两层 Oracle；
3. 将状态排序、问题级停止风险、端到端质量和成本分开；
4. 将失败分解为证据缺失、候选缺失、Critic 误判、错误提前越界和预算耗尽。

本文不主张：

- 提出新的 Search-R1 或 SIM-RAG 算法；
- 重新训练 adapted Critic 本身构成算法创新；
- 对所有多轮 RAG、开放 Web Search 或所有数据分布给出普遍结论；
- answer-event veto 等价于任意时刻的完整最优停止；
- 高 AUC、整体 EM 或证据完整率可以替代 accepted-risk 评价。

## 8. 结果分支与对应结论

### Dense Oracle 仍低

当前接口没有足够可选择余量，不启动大规模 adapted Critic 训练。论文转向
“停止优化的可辨识条件”，重点分析证据已充分但候选仍错误、额外搜索无
收益和 Native 已捕获全部可达候选。

### Released 失败、adapted 成功

通用 Critic 不能直接跨 Reasoner 和轨迹格式迁移；目标轨迹适配是获得有效停止信号的关键。

### Released 与 adapted 均失败、Oracle 较高

即使上游系统可信且 Critic 与目标轨迹匹配，状态判别仍未转化为有效的首次越界停止，问题主要位于停止建模或接口语义。

### 稀疏 Oracle 低、Dense Oracle 较高

answer-event veto 低估了停止余量；逐轮答案构造是答案级停止研究的必要
接口条件，但其额外 LM calls、token 和延迟必须计入成本。

### SIM-RAG 提高整体质量但未达到风险目标

性能增强不等于可认证的低风险停止；EM、AUC 和 accepted risk 必须分别报告。

### Adapted SIM-RAG 同时提高质量并达到风险目标

在可信上游、目标轨迹适配和独立风险校准共同成立时，可以把状态级 Critic 转化为具有有限样本风险证据的停止策略。

### 扩充版 S2G Judge：Gate S5 非劣且减少搜索

Gate S5 的两道预注册系统门均已通过。可报告为：轻量结构化 Judge 在
untouched suffix-800 上，将部分状态排序信号转化为平均 0.09625 次/题
的搜索减少，同时 Official EM 差的 95% CI 下界仍高于 −2pp。必须同时
报告 EM 点估计为负、固定阈值 STOP precision 为 0.6216，以及该结论只
适用于冻结的 Search-R1/HotpotQA 配置。

## 9. 建议标题

中文暂定：

> 多轮 RAG 何时停止？Search-R1 中的结构化停止判断与检索节省

英文暂定：

> When Should Multi-Round RAG Stop? Structured Stopping Judgments and Retrieval Reduction in Search-R1

## 10. 稿件处理原则

- 早期 BM25 闭集、preliminary checkpoint、MuSiQue 和旧 SIM-RAG 迁移结果全部标记为 `legacy/pilot`。
- 旧结果不进入新论文正文或附录，也不用于选择新阈值、数据子集或报告口径。
- 原始文件保留在仓库中，用于审计和生成旧问题 ID 排除清单。
- 新摘要、主结果和结论只依据新实验。
- 关于 Search-R1 基线只使用 `CONFIGURATION-VERIFIED/COMPATIBLE` 或“官方配置已核验且与论文报告值统计相容”的措辞，不使用 `REPRODUCED`。
