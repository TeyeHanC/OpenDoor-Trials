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
.venv/bin/python -m trialmatch run --patient trec2021-1 --nct NCT00002814 --trial-source snapshot
.venv/bin/python -m trialmatch run --patient sample-nsclc --nct NCT04035486
```

## 数据

| 数据 | 来源 |
| --- | --- |
| 试验（演示和产品链路） | ClinicalTrials.gov API v2，按 NCT 编号拉取当前注册信息，缓存在 `data/cache/trials/` |
| 试验（评测） | TREC 标注时使用的 2021-04-27 注册快照，`--trial-source snapshot` |
| 患者 | 包内自带的合成病历 `sample-*`，或 TREC Clinical Trials 2021/2022 的患者描述 `trec2021-N` |
| 评测标注 | TREC qrels：0 不相关，1 被排除，2 符合 |

患者来源在代码里是白名单（`trialmatch/patients.py`），没有读取任意文件的入口，因此不会把 MIMIC 等受限数据或真实患者信息发到外部 API。`data/` 目录不进 git。

## 处理流程

1. **标准拆解**（`decompose.py`）：模型把入排标准原文拆成原子条件，保留原文、类型、时间窗、数值阈值；医学概念给出标准名，没把握时留空。年龄、性别这类只存在于注册字段里的条件由程序补上。按试验缓存。
2. **证据定位**（`segment.py`、`retrieval.py`）：病历按段落切分并记录字符偏移。病历不超过 `TRIALMATCH_FULL_NOTE_MAX_CHARS`（默认 20000 字符，约 5000 token）时不做检索，全文交给模型；超过才用 BM25 为每条条件取 top-k 段落。BM25 是纯词面匹配，对不上"肺腺癌"和"NSCLC"这类同义表述，漏检会被误判成"信息缺失"，所以只作为超长病历的退路。
3. **逐条判定**（`judge.py`）：每条输出 `met` / `not_met` / `unknown`，附引用、一句理由、置信度。模型还要自报 `basis`：`stated` 表示引用本身就在说这条条件问的事，`inferred` 表示结论是从年龄、性别这类明确事实推出来的（只允许这一类推论）。
4. **引用校验**（`verify.py`）：程序在病历原文里逐字查找每条引用（只容忍空白字符差异），偏移由程序计算，不采信模型给的位置。找不到任何有效引用的"符合 / 不符合"会被降级为"信息缺失"并标记。
5. **汇总**（`summarize.py`）：按规则得出整体结论，不再调用模型，所以结论可以从条件列表手工复算。

### 程序复算的算术

涉及时间窗或数值阈值的条件，模型除了给出判定，还要填一份"核对单"：它从病历读到的日期或数值、条件里的窗口或阈值，以及它自己算出的答案。程序用这些输入重新算一遍（`checks.py`）：

| 结果 | 处理 |
| --- | --- |
| 与模型一致 | 记为已复算，命令行显示算式 |
| 与模型不一致 | 判定降为"信息缺失"，标记 `calculation_mismatch`，交人工核对 |
| 无法复算（日期不完整、单位不认识） | 保留判定，置信度降为低，标记 `calculation_not_checkable` |
| 输入的日期或数值在病历原文里找不到 | 保留判定，置信度降为低，标记 `calculation_input_not_in_note` |

支持回溯时间窗（月、年按日历计算）、数值比较（含"2.5 倍正常上限"和区间）、常见检验单位换算。这不依赖模型的工具调用能力，所以换模型、换网关都能用，而且计算一定会执行。

### 工具

模型只在两处调用工具，定义和实现在 `trialmatch/tools.py`，输入用严格 schema 约束，出错时把错误信息还给模型而不是中断流程。

| 工具 | 用在哪里 | 作用 |
| --- | --- | --- |
| `search_note`、`read_segments` | 判定，仅超长病历 | 按关键词搜病历、按段号读上下文，弥补 BM25 预取漏掉同义表述的问题 |
| `search_trials`、`get_trial_criteria` | 找试验 | 检索 ClinicalTrials.gov、读某个试验的入排标准 |

病历能整篇发给模型时（默认 20000 字符以内），判定环节不带任何工具。超长病历的搜索调用记录在结果的 `tool_calls` 里；`TRIALMATCH_JUDGE_TOOLS=0` 可以关掉。`tools.py` 里的日期、阈值、换算函数现在由 `checks.py` 直接调用，也仍可作为工具供 `check-model` 自检使用。

### 为患者找试验

```bash
.venv/bin/python -m trialmatch find-trials --patient sample-nsclc --max 5
.venv/bin/python -m trialmatch find-trials --patient sample-nsclc --match   # 对入围的每个试验再跑逐条判定
```

这是唯一由模型自己决定步骤的环节（`find.py`）：它根据病历决定搜什么、看结果、换说法再搜、读候选试验的标准，最后给出入围名单和每个试验"为什么相关 / 还要核对什么"。入围名单不是判定，逐条判定仍由上面的流水线完成。

- 程序只保留工具实际返回过的 NCT 编号；模型凭记忆报出的编号会被丢弃并记在 `rejected_ids` 里。
- 检索词会发到 ClinicalTrials.gov。提示词要求只发临床术语，但这一条程序无法强制，所以这个命令和其他命令一样只接受合成或公开病历。
- 结果存在 `data/find/<患者>.json`。目前只有命令行入口，界面里还没有。

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

每条条件的输出字段见 `trialmatch/models.py` 的 `CriterionResult`，包括 `criterion_id`、`criterion_text`、`source_text`、`type`、`verdict`、`evidence[{quote,start,end,segment_id}]`、`rationale`、`confidence`，`basis`，以及审计字段 `model_verdict`、`downgraded`、`rejected_quotes`。

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
- `errors.csv` / `errors.jsonl`：判错的案例，漏掉的合格患者排在最前，每行带一个在界面里打开该案例的链接；`inferred_criteria` 列列出靠推论得出的条件
- `error_criteria.csv`：判错案例里每条"符合 / 不符合"判定一行，`used_inference` 列标出该条是否用了推论而没有直接引用
- `predictions.csv`、`failures.json`

金标准正例是 TREC 标注 2（符合）。系统有三种结论，所以正例预测报告两种口径：

- **screen-in（主指标）**：没有被判为 `not_eligible` 就算推荐。只有系统明确判"不符合"才算漏掉患者。
- **strict（辅助）**：只有 `possibly_eligible` 算推荐。

screen-in 召回率单独看没有意义（什么都不排除也是 100%），所以总是和两个数一起报告：TREC 标为"被排除"的配对中被判为 `not_eligible` 的比例，以及 `needs_more_info` 占全部配对的比例。

评测默认用 TREC 快照里的入排标准（`--trial-source snapshot`），保证和标注时一致。快照是五个 zip 共约 1.8 GB，程序用 HTTP 范围请求只取需要的试验；首次使用会花约 20 秒建立索引（`data/trec/corpus/index.json`）。如果把五个 zip 下载到 `data/trec/corpus/`，会改为读本地文件。快照的评测结果单独存为 `…__snap2021`，不会和通过 API 跑出的结果混用。

已跑过的患者-试验对会复用 `data/runs/` 里的结果，评测中断后重跑同一条命令即可续跑；`--rerun` 强制重新调用模型。

### 每批判定条数的对比

默认每次请求判 8 条条件。下面的命令对同一个患者-试验配对分别按"每批 8 条"和"每条单独判"各跑一次，输出判定不一致的条件清单，两次结果都会保存，可以在界面里打开：

```bash
.venv/bin/python -m trialmatch compare-batching --patient sample-nsclc --nct NCT04035486
```

模型输出本身不是确定性的，同一设置跑两次也可能有个别差异，单条不一致只能当线索。

### 已知局限

- `basis` 是模型自报的，程序无法校验；它用来帮人定位要重点检查的判定，不是保证。
- 均衡抽样下的精确率不代表真实患病率下的精确率，可用 `--sampling proportional` 对比。
- TREC 患者描述很短，大多数条件会是"信息缺失"，strict 口径的召回率预期很低。
- 模型若拒答某一批条件，这些条件记为"信息缺失"并打上 `model_declined` 标记。
- 工具链路（判定时调用计算器、找试验的搜索循环）只用假模型和本地抓包验证过请求格式，还没有真实 API 结果；模型实际会不会用、用得对不对未知。
- 程序只复算模型交上来的输入；模型如果读错了病历里的数值，复算会"一致"但结论仍是错的。输入不在原文里的情况会被标记，读错成另一个真实出现的数字则抓不到。
- 模型不填核对单时程序无从复算，这一点靠提示词要求，无法强制。

## 开发

```bash
make test
```

测试用脚本化的假模型，不调用 API。配置项（模型、effort、数据目录）见 `.env.example`。
