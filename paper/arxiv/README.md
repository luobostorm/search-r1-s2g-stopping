# arXiv 论文工程

本目录是“多轮 RAG 何时停止”论文的投稿源码。2026-08-04 起，旧的
BM25/MuSiQue/CP 风险认证稿退出主线；当前论文以 Search-R1 可达性审计、
S2G 机制实验和 suffix-800 确认性结果为核心。

## 稿件

- `main-en.tex`：英文 arXiv 主稿。
- `main.tex`：中文逐项复核稿，不作为独立科学结果源。
- `references.bib`：仅保留新主线实际使用并已核验的一手文献。
- `figures/`：主线三张投稿图、PNG 预览与 source/hash manifest。
- `../../scripts/generate_s2g_paper_figures.py`：从冻结 JSON 证据重建图表。
- `Makefile`：静态检查、Tectonic 编译与隔离构建入口。

当前标题：

- 中文：多轮 RAG 何时停止？候选可达性、策略风险与检索成本
- 英文：When Should Multi-Round RAG Stop? Candidate Reachability, Policy Risk, and Retrieval Cost

## 主张边界

确认性主分析仅使用 HotpotQA indices 200--999。论文主线应表述为：本文把
S2G 式结构化 Judge 机制适配到冻结的 Search-R1，并基于 Search-R1 状态
专门训练 Qwen3.5-2B Judge。实验开始前规定 Official EM 最多允许下降 2 个
百分点；实际差为 `-0.00625`，95% CI `[-0.01250, 0]`，下降未超过这一
可接受范围。与此同时，平均搜索差为 `-0.09625`，95% CI
`[-0.12000, -0.07375]`，即少 77 次检索、平均减少 3.70%。

论文不得写成答案质量提高、停止安全、风险认证、Pareto 支配或总体推理效率
提高。扩充策略的 69 次提前停止中有 27 次不安全，状态级 STOP precision
为 0.6216，必须与搜索节省同时出现。

## 编译与检查

```bash
cd paper/arxiv
make figures
make check
make draft-pdf
make check-isolated
```

英文投稿 PDF 输出到 `output/pdf/main.pdf`；中文复核 PDF 输出到
`output/pdf-zh/main.pdf`。源码使用 XeTeX/Tectonic、BibTeX 与 `natbib`。
`make figures` 会直接读取 H0 与 suffix-800 的冻结 JSON；定量图不以论文
正文中的手抄数字为数据源。

静态检查会拒绝未定义引用、LaTeX 环境或花括号不配对、Markdown 残留，
以及旧稿的 formal-result marker。隔离构建只复制本目录的论文源码，检查
不存在本机绝对路径依赖。

## 本公开快照中的权威证据

- `results/s2g-scaleup-aligned-eval-v1/final-dev1000-v2/evaluation-confirmatory-suffix800.json`
- `results/searchr1-v02-reproduction/gateh0/quality-cost-report.json`
- `results/training-summary.json`
- `docs/多轮RAG何时停止-论文核心立意.md`
- `docs/S2G规模扩充与同测试集评测实验协议.md`

任何数字修订都应先修改权威结果源或追加勘误，再同步两份 LaTeX；不得只在
论文中手工改变科学结果。
