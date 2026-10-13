# OpenDoor Trials：临床试验患者匹配原型

读入一份患者病历和一个试验的入排标准，逐条判断患者是否符合，每个判断都附病历原文证据，最后由医生复核。

**本系统只做辅助筛选，不做入组决定，所有判断须由医生最终确认。**

## 快速开始

```bash
cp .env.example .env          # 填入 ANTHROPIC_API_KEY
make demo                     # 建虚拟环境、安装依赖、启动 http://127.0.0.1:8000
```

在页面顶部选一个患者、填 NCT 编号、点 Run match。也可以用命令行：

```bash
.venv/bin/python -m trialmatch fetch-trec --year 2021          # 下载 TREC 患者和标注
.venv/bin/python -m trialmatch run --patient trec2021-1 --nct NCT00002814
.venv/bin/python -m trialmatch run --patient sample-nsclc --nct NCT04035486
```

## 数据

| 数据 | 来源 |
| --- | --- |
| 试验 | ClinicalTrials.gov API v2，按 NCT 编号拉取，缓存在 `data/cache/trials/` |
| 患者 | 包内自带的合成病历 `sample-*`，或 TREC Clinical Trials 2021/2022 的患者描述 `trec2021-N` |
| 评测标注 | TREC qrels：0 不相关，1 被排除，2 符合 |

患者来源在代码里是白名单（`trialmatch/patients.py`），没有读取任意文件的入口，因此不会把 MIMIC 等受限数据或真实患者信息发到外部 API。`data/` 目录不进 git。

## 处理流程

1. **标准拆解**（`decompose.py`）：模型把入排标准原文拆成原子条件，保留原文、类型、时间窗、数值阈值；医学概念给出标准名，没把握时留空。年龄、性别这类只存在于注册字段里的条件由程序补上。按试验缓存。
2. **证据检索**（`segment.py`、`retrieval.py`）：病历按段落切分并记录字符偏移，BM25 给每条条件的相关段落排序。病历不超过 6000 字符时全部段落都交给模型，避免漏检被误判成"信息缺失"；更长的病历只送 top-k。
3. **逐条判定**（`judge.py`）：每条输出 `met` / `not_met` / `unknown`，附引用、一句理由、置信度。
4. **引用校验**（`verify.py`）：程序在病历原文里逐字查找每条引用（只容忍空白字符差异），偏移由程序计算，不采信模型给的位置。找不到任何有效引用的"符合 / 不符合"会被降级为"信息缺失"并标记。
5. **汇总**（`summarize.py`）：按规则得出整体结论，不再调用模型，所以结论可以从条件列表手工复算。

### 判定语义

`verdict` 表示患者是否满足这条条件的**字面表述**，`effect` 表示它对入组的影响：

| 类型 | verdict | effect | 界面显示 |
| --- | --- | --- | --- |
| 入选 | met | supports | ✓ Met |
| 入选 | not_met | blocks | ✕ Not met — blocks enrollment |
| 排除 | met | blocks | ✕ Exclusion applies — blocks enrollment |
| 排除 | not_met | supports | ✓ Exclusion does not apply |
| 任意 | unknown | unresolved | ? Insufficient information |

整体结论：有任何 blocks 为 `not_eligible`；否则有任何 unresolved 为 `needs_more_info`；否则 `possibly_eligible`。

每条条件的输出字段见 `trialmatch/models.py` 的 `CriterionResult`，包括 `criterion_id`、`criterion_text`、`source_text`、`type`、`verdict`、`evidence[{quote,start,end,segment_id}]`、`rationale`、`confidence`，以及审计字段 `model_verdict`、`downgraded`、`rejected_quotes`。

## 界面

左边病历全文，右边条件列表，按对入组的影响着色。点击条件，病历滚动并高亮它的证据。低置信度和引用未通过校验的条目默认排在最前。每条有 Agree / Modify，医生的每次操作追加写入 `data/reviews.jsonl`（不覆盖历史），顶部会显示复核后的结论。

```bash
.venv/bin/python -m trialmatch review-stats    # 医生与系统的一致率
```

## 评测

```bash
.venv/bin/python -m trialmatch eval --dry-run            # 只看抽样，不调用模型
.venv/bin/python -m trialmatch eval --n 10               # 先小规模试跑，看实际 token 用量
.venv/bin/python -m trialmatch eval --n 500              # 默认：三类标注各取约 1/3，固定随机种子
```

输出在 `data/eval/<name>/`：

- `metrics.json`：患者-试验层面的召回率和精确率、标注 × 结论的混淆表、引用校验通过率
- `errors.csv` / `errors.jsonl`：判错的案例，漏掉的合格患者排在最前，每行带一个在界面里打开该案例的链接
- `predictions.csv`、`failures.json`

金标准正例是 TREC 标注 2（符合）。系统有三种结论，所以正例预测报告两种口径：

- **screen-in（主指标）**：没有被判为 `not_eligible` 就算推荐。只有系统明确判"不符合"才算漏掉患者。
- **strict**：只有 `possibly_eligible` 算推荐。

已跑过的患者-试验对会复用 `data/runs/` 里的结果，评测中断后重跑同一条命令即可续跑；`--rerun` 强制重新调用模型。

### 已知局限

- API 返回的是试验**当前**的注册信息，可能与 TREC 标注时（2021 年 4 月快照）的入排标准不同。
- 均衡抽样下的精确率不代表真实患病率下的精确率，可用 `--sampling proportional` 对比。
- TREC 患者描述很短，大多数条件会是"信息缺失"，strict 口径的召回率预期很低。
- 模型若拒答某一批条件，这些条件记为"信息缺失"并打上 `model_declined` 标记。

## 开发

```bash
make test
```

测试用脚本化的假模型，不调用 API。配置项（模型、effort、数据目录）见 `.env.example`。
