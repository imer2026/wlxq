# 英雄格分类器使用手册

本文档面向实际采集、挑图、训练和验证人员。正式流程是：原始截图按局保存；选定多局创建一个 `train`、`validation` 或 `test` import；程序把这一批局集中裁切、联合聚类；人工从相似图片簇中挑图并移动到 split 共用的 `labeled/`；最后由脚本重建清单。

> **前置门禁已经完成（2026-08-11）。** helper 棋盘以排 2～排 6 的 10 个实测格中心拟合参数，并通过带裁剪框和标签的预览图逐格确认全部 12 格；1A、2A 使用确认后的等距模型定位。可以直接开始正式裁切、标注和训练。

## 最小可行性验证

先打 4 局即可判断路线是否可行：2 局 train、1 局 validation、1 局 test。4 局只能验证采集、裁切、人工挑图、训练、跨局验证和运行速度是否走得通，不能证明模型已经达到正式使用精度。

正式首批建议 10 局：6 局 train、2 局 validation、2 局 test。后续继续增加数据时，为 train 新建 import 即可，旧的 `labeled/` 不会被重新聚类或覆盖。

## 安装和另一台电脑的条件

采集与裁切电脑安装开发环境：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

训练电脑额外安装：

```powershell
pip install -e ".[train]"
```

默认训练使用 MobileNetV3-Small 预训练权重，首次可能联网下载。离线时可用 `--no-pretrained`，但效果必须重新评估。2020 款 MateBook X Pro（i7、16 GB、SSD、MX250）可训练；若 PyTorch CUDA 与显卡驱动匹配，程序会自动使用 CUDA，不存在项目内的“开启 GPU”开关。CUDA 不可用时自动使用 CPU。

如果另一台电脑只负责训练已裁切、已标注数据，不需要游戏窗口、`configs/local.yaml` 或截图环境，只需：

- 同一版本项目代码及 Python 3.10+ 环境；
- 对应数据组中的 `train/labeled`、`validation/labeled` 和清单；
- 首次下载预训练权重所需网络，或已有缓存；
- 足够磁盘空间和可写的模型输出目录。

数据和模型默认属于本地生成物，不提交 Git。跨电脑时使用移动硬盘、局域网或私有文件存储复制。

## 固定目录结构

以强袭主 C、3000×2000 显示器、helper 为例，数据组根目录为：

```text
datasets/hero_classifier/assault/3000x2000/helper/
```

完整结构：

```text
helper/
  rounds/
    202608111914/
      raw/
        202608111914_frame000001.png
      capture_manifest.csv
    202608112021/
      raw/
      capture_manifest.csv

  train/
    labeled/                         # 永久累积；新增 import 不会改这里
      assault/star1/
      assault/star2/
      angel/star1/
      empty/plain/
      empty/effect/
      unavailable/plain/
      unavailable/effect/
      unknown/
    imports/
      20260812_001/
        rounds.txt                   # 本批来源局，由脚本写
        unclassified/               # 本批多局裁切后联合聚类
          000_x0850_c12/
          001_x0632_c12/
        candidates/                 # 人工只需重点查看这里
          000_x0850_c12/
            000_x0010/
          001_x0632_c12/
            000_x0010/
            001_x0003/
        manifest.csv                 # 本批裁切追溯清单
        candidate_manifest.csv       # 候选图到原图的追溯清单
        suggested/                   # 可选；模型建议副本，需人工确认
        prediction_manifest.csv      # 可选；逐图预测和小组决策追溯清单
      20260815_001/                  # 后续新增局的新批次
        rounds.txt
        unclassified/
        candidates/
        manifest.csv
        candidate_manifest.csv
    dataset_manifest.csv             # sync-labels 完整重建

  validation/
    labeled/
    imports/
    dataset_manifest.csv

  test/
    labeled/
    imports/
    dataset_manifest.csv
```

原始截图始终按局保存在 `rounds/<round_id>/`。裁切图按 split 和 import 聚合；同一 import 中的多局只联合聚类一次。程序再从每个一级簇生成少量 `candidates/`，原图仍完整保留在 `unclassified/`。train、validation、test 不混合聚类，同一局也只能登记到其中一个 split。

文件名：

```text
202608111914_frame000123_4B.png
```

- `202608111914`：对局开始时间；
- `frame000123`：本局第 123 个采集时间点；
- `4B`：第 4 行 B 列。棋盘只有 3 列、6 行。

文件名不能在人工分类时修改，因为脚本依靠它恢复来源局、帧和格子。

## 一、局内采集

先确保游戏窗口、项目配置和 12 格坐标对应当前环境，然后在一局开始后执行：

```powershell
wlxq-bot hero-classifier collect --main-c assault --role helper
```

默认每秒计划采集 1 张，持续 360 秒。截图线程只抓完整客户区并放入保存队列，后台线程异步编码 PNG，因此不会要求一秒内同步完成截图、12 格裁切和写完全部文件。机器繁忙时可能产生队列丢弃或调度跳过，最终以 `capture_manifest.csv` 为准。

默认输出：

```text
datasets/hero_classifier/assault/<显示器分辨率>/helper/rounds/<YYYYMMDDHHMM>/
```

也可指定：

```powershell
wlxq-bot hero-classifier collect --main-c assault --role helper --duration 360 --interval 1 --round-id 202608111914
```

采集期间不要调整窗口大小、DPI、显示器或角色。采到结算页的帧后续不能标成空格，应标为 `unknown` 或留在 `unclassified`。

## 二、按多局创建 import 并联合裁切

不再逐局执行 crop 作为正式流程。先打完并保留若干局，再一次指定到某个 split：

```powershell
$root = "datasets/hero_classifier/assault/3000x2000/helper"

wlxq-bot hero-classifier import-rounds $root `
  --split train `
  --rounds 202608111914,202608112021,202608121101,202608121209,202608131903,202608132012 `
  --import-id 20260814_001

wlxq-bot hero-classifier import-rounds $root `
  --split validation `
  --rounds 202608141850,202608141959 `
  --import-id 20260814_001

wlxq-bot hero-classifier import-rounds $root `
  --split test `
  --rounds 202608151906,202608152015 `
  --import-id 20260815_001
```

`import-rounds` 会：

1. 检查每个 `rounds/<局号>/raw/` 存在；
2. 扫描三个 split 的全部历史 import；已经完成裁切和聚类的局提示其原 split/import 并跳过；
3. 把本批剩余的新局裁到同一个 `imports/<import_id>/unclassified/`；
4. 所有新局裁完后使用阈值 35 联合一级聚类；
5. 为所有一级簇生成候选：不超过 100 张直接挑最多 10 张；超过 100 张先用阈值 15 二次细分，每个二级簇挑最多 10 张；
6. `rounds.txt` 只写实际新处理的局，并生成 `manifest.csv` 和 `candidate_manifest.csv`；
7. 只补齐 split 的标签目录，不移动、不删除、不覆盖旧 `labeled/` 图片；
8. 如果请求中的局全部处理过，只输出跳过提示，不创建空 import。

例如第一次处理 `a/b/c/d`，第二次请求 `d/e/f/g`：第二次会提示 `d` 已在原 import 完成，只对 `e/f/g` 裁切和联合聚类。这个门禁以历史 `imports/*/rounds.txt` 为准，不依赖人工记忆。

裁切一般远快于一局时长，主要耗时是 PNG 解码、12 格编码和聚类。SSD、4 个 worker 下通常以分钟计；实际耗时以命令输出为准。

长流程不会一直无输出。每条 INFO 日志前会显示当前本地输出时间；阶段开始和完成必定输出。耗时超过约 5 秒时会周期性显示已处理/总数、百分比、已耗时，以及当前簇数、候选组数等指标。可见阶段包括逐局裁切、阈值 35 一级聚类、聚类结果整理、阈值 15 大簇二次细分和 candidates 生成。示例：

```text
14:32:10 一级聚类 进度 processed=7200/18000 percent=40.0% elapsed=83.2s clusters=37
14:34:51 候选生成 进度 processed=9500/18000 percent=52.8% elapsed=31.4s candidate_groups=42 candidate_images=386
```

`crop` 和 `group` 命令仍保留用于单局排查或旧数据兼容，但不用于上述增量主流程。

如果历史 import 已经有 `unclassified/`，但创建时尚未支持 candidates，不需要重新裁切或重做一级聚类，可执行：

```powershell
wlxq-bot hero-classifier select-candidates `
  "$root/train/imports/20260812_001"
```

该命令使用同样的 `100 / 15 / 10` 默认规则补生成 `candidates/` 和 `candidate_manifest.csv`。已经存在 candidates 时会拒绝覆盖，避免破坏已进行的人工分类。

## 三、人工挑图和分类

自动聚类和 candidates 都只是人工挑图辅助，不识别英雄、不自动判断星级、不自动贴标签，也不是最终训练数据。一级聚类把大量散图整理成相似图片簇，候选生成再从中挑出少量尽量不重复、尽量覆盖视觉变化的图片。

日常标注只需打开当前 import 的 `candidates/`。最内层目录对应一个最终候选小组：

```text
candidates/
  000_x1850_c12/             # 对应阈值35生成的一级簇
    000_x0010/               # 阈值15二级簇选出的10张候选
    001_x0004/
  001_x0085_c06/             # 一级簇不超过100张，不二次细分
    000_x0010/               # 直接从一级簇挑出的10张候选
```

规则固定为：

- 一级聚类阈值默认 `35`；
- 一级簇 `≤100` 张：不二次细分，默认直接挑最多 `10` 张；
- 一级簇 `>100` 张：使用阈值 `15` 二次细分，每个二级簇默认挑最多 `10` 张；
- 候选先取稳定的第一张，之后优先选择与已有候选视觉差异最大的图片，避免连续近似帧占满名额；
- candidates 使用复制文件，移动候选不会破坏 `unclassified/` 原图；
- 候选小组仍不保证标签一定一致，必须人工确认。

候选上限已经支持配置。新 import 使用 `--candidate-max-per-group`，历史 import 使用 `select-candidates --max-per-group`。一般保持默认 10；如果实测仍担心遗漏，可以提高，但候选目录和人工量也会同步增加。

确认后把 candidates 中的 PNG 移动到同一 split 的集中目录：

```text
<split>/labeled/
  <hero>/star1/
  <hero>/star2/
  <hero>/star3/
  <hero>/star4/
  empty/plain/
  empty/effect/
  unavailable/plain/
  unavailable/effect/
  unknown/
```

规则：

- 英雄和星级能够可靠确认：`labeled/<英雄>/star<N>/`；
- 英雄被强袭或其他单位部分遮挡，但英雄和星级仍能可靠确认：仍放真实英雄的 `star<N>/`；
- 没有英雄、没有明显特效：`empty/plain/`；
- 空格上有技能、弹道、光效等：`empty/effect/`；
- 逻辑位置尚未开放：`unavailable/plain/` 或 `unavailable/effect/`；
- 遮挡严重、英雄/星级不确定或不是有效棋盘画面：`unknown/`。

如果最内层候选目录中的图片全部为同英雄同星级，可以一次选择其中全部 PNG 移入对应标签目录；混有不同标签时分别移动。`unclassified/` 主要用于追溯或候选不足时补查，通常不需要逐张浏览。

第一版不使用 `clean/occluded`。可确认的普通图和遮挡图放在同一个英雄星级目录；无法确认才进 `unknown`。重复和相似图片允许保留。训练器先按主 C、星级和负样本类别应用业务权重，再在类别内部按来源局和样本类型均衡，避免图片多的类别或某一局仅凭原始数量压过其他数据。

## 四、同步标签清单

人工只移动图片，不手工维护 CSV。每次挑图结束后执行：

```powershell
wlxq-bot hero-classifier sync-labels $root --split train
wlxq-bot hero-classifier sync-labels $root --split validation
wlxq-bot hero-classifier sync-labels $root --split test
```

脚本扫描当前 `labeled/` 并完整、原子重建 `dataset_manifest.csv`，不是追加更新。因此图片后续被移动或删除，再同步时清单会准确反映目录现状。脚本还会校验：

- 文件名仍符合约定；
- 来源局已登记到当前 split；
- 来源局真实存在；
- 同一来源图没有被重复放入多个标签目录；
- 标签目录只使用允许的层级。

`unknown` 会记录到数据清单以便追溯和后续拒绝策略验证，但不会作为普通训练类别读取。

## 五、后续新增局

新增局仍保存在 `rounds/`，然后创建新的 import：

```powershell
wlxq-bot hero-classifier import-rounds $root `
  --split train `
  --rounds 202608201901,202608202012 `
  --import-id 20260820_001
```

程序只创建 `train/imports/20260820_001/`，不会重新聚类旧 import，也不会改 `train/labeled/`。挑完新批图片后再次执行 `sync-labels`，清单会扫描旧标签与新标签并完整重建。

validation 和 test 也允许后续增加 import，但要保持评估纪律：如果根据 test 结果调整了模型、阈值或数据，原 test 已参与决策，不再是最终独立测试，应另采新的 test 局。

## 六、训练

确认 train 和 validation 都完成挑图及 `sync-labels` 后执行：

```powershell
wlxq-bot hero-classifier train $root `
  --output outputs/hero_classifier/assault-helper
```

命令自动读取 `$root/train/labeled` 和 `$root/validation/labeled`，不再手工传 `--train-rounds`、`--validation-rounds`。

标准数据组目录是 `<主C>/<分辨率>/<角色>`，因此上例会自动从 `$root` 推导主 C 为 `assault`；使用非标准目录时必须显式传 `--main-c assault`。训练默认对每一个实际存在的类别应用以下相对抽样权重：

```text
每个1星英雄类别       1.0
每个2星英雄类别       1.0
主C 3星               0.8
其他英雄3星           0.5
主C 4星               0.3
其他英雄4星           0.1
empty                  0.5
unavailable            0.3
```

这些数字是类别的相对抽样权重，不是要求标签图片达到对应数量比例。类别内部仍按来源局均衡；`empty` 和 `unavailable` 内部再按 `plain/effect` 均衡。某个英雄没有 4 星图片时，该类别不会出现在模型中，也不会报错或要求补齐。实际存在的类别及最终权重会写入 `hero_classifier.json` 的 `class_sampling_weights`。

一般直接使用默认值即可。需要做对照实验时可以覆盖：

```powershell
wlxq-bot hero-classifier train $root `
  --main-c assault `
  --star1-weight 1.0 `
  --star2-weight 1.0 `
  --main-c-star3-weight 0.8 `
  --other-star3-weight 0.5 `
  --main-c-star4-weight 0.3 `
  --other-star4-weight 0.1 `
  --empty-weight 0.5 `
  --unavailable-weight 0.3 `
  --output outputs/hero_classifier/assault-helper
```

所有权重必须大于 0。降低权重表示减少抽样机会，不等于删除该类别。

训练产物：

```text
outputs/hero_classifier/assault-helper/
  hero_classifier.pt
  hero_classifier.onnx
  hero_classifier.json
  training_history.json
```

16 GB 内存、i7、SSD、MX250 的笔记本，首批 6+2 局挑出的中小规模数据，训练加每轮 validation 通常约 30～90 分钟；CUDA 可用时一般更快，CPU 可能更久。这个区间包括训练过程中的 validation，不包括后面的独立 test 评估和人工标注时间。

正式运行只加载 ONNX，不需要 PyTorch。训练完成后，把 `configs/default.yaml` 中对应主 C 的 `main_c_profiles.<主C>.hero_classifier_model` 指向导出的 `hero_classifier.onnx`，并保留同目录同名的 `hero_classifier.json`。合作任务每帧按已标定 ROI 裁出 12 格并一次批量推理，再对多帧精确类别做过半投票；不再使用英雄模板匹配作为棋盘识别回退。单次批量识别通常为毫秒到几十毫秒量级，应在目标电脑实测。

## 七、用已训练模型预分类新 candidates

初始模型通过独立验证后，可以对后续新增 import 的 candidates 做离线预分类。该功能不连接游戏窗口、不做正式棋盘识别，只减少人工逐组判断的工作量：

```powershell
$import = "$root/train/imports/20260820_001"
$model = "outputs/hero_classifier/assault-helper/hero_classifier.onnx"

wlxq-bot hero-classifier suggest-labels $import `
  --model $model
```

默认读取 ONNX 同目录同名的 `hero_classifier.json`，使用其中的 confidence 和 margin 门槛。需要临时采用更严格的人工预分类门槛时，可以显式覆盖：

```powershell
wlxq-bot hero-classifier suggest-labels $import `
  --model $model `
  --confidence-threshold 0.90 `
  --margin-threshold 0.30
```

命令只读取：

```text
<import>/candidates/
<import>/candidate_manifest.csv
```

生成：

```text
<import>/
  suggested/
    assault_star2/
      <一级簇>/<二级簇>/*.png
    angel_star1/
      <一级簇>/<二级簇>/*.png
    empty/
      <一级簇>/<二级簇>/*.png
    unavailable/
      <一级簇>/<二级簇>/*.png
    review/
      low_confidence/          # 至少一张低 confidence 或低 margin
        <一级簇>/<二级簇>/*.png
      mixed_group/             # 同组候选预测为多个不同的精确类别
        <一级簇>/<二级簇>/*.png
      unknown/                 # 无法使用的预测
        <一级簇>/<二级簇>/*.png
  prediction_manifest.csv
```

一个小组只有在“每张图片预测为同一个精确类别（包括星级），并且每张都通过 confidence 和 margin 门槛”时，才进入普通 `suggested/<类别>/`。例如 `assault_star1` 和 `assault_star2` 属于组内冲突，会整组进入 `review/mixed_group/`。低 confidence 与低 margin 都进入 `review/low_confidence/`。

`prediction_manifest.csv` 记录原 candidate 清单字段、模型路径和 SHA-256、实际门槛、top1/top2 类别和概率、margin、单图拒绝原因、小组决策及建议副本路径，方便追溯。`suggested/` 中的图片是副本；原 `candidates/` 和 `unclassified/` 保持不变。已有 `suggested/` 或 `prediction_manifest.csv` 时命令拒绝覆盖，避免破坏正在进行的人工审核。若确需用另一个模型重跑，应先人工确认旧结果是否还需要保留，再明确备份或处理旧输出。

人工操作仍然是最终门禁：

1. 打开 `suggested/`，确认模型建议是否正确；
2. 把确认无误的 **PNG 文件** 移到当前 split 的 `labeled/<英雄>/star<N>/`；不要把一级簇或二级簇目录整体移进去；
3. `empty` 和 `unavailable` 仍需人工区分 `plain/effect`；模型只预测大类；
4. 错误建议按真实类别移动，无法确认的图片移到 `labeled/unknown/`；
5. 最后运行 `hero-classifier sync-labels`。

模型建议不会直接进入训练数据，也不能把一个 candidates 小组的结论扩散为整个 `unclassified` 一级簇的标签。这样可以避免错误伪标签在重新训练时不断自我强化。

## 七点五、种子模型与主动学习循环的实战经验

以下经验来自 assault/helper 数据组的实际迭代（种子 1598 张起步，两轮循环扩到约 9000 张，validation 从 94.5% 升至 96%+）。核心流程：

```text
手标少量局（尽量覆盖全部 英雄×星级）
  -> 训练种子模型 -> suggest-labels 预分类大批新局
  -> 高置信度建议人工抽查确认后搬入 labeled -> sync-labels -> 重训
  -> 新模型更强 -> 处理下一批 -> 循环
```

### 种子模型的第一要求是类别完备，不是单类数量

模型的输出类别来自 train/labeled 里实际存在的类。缺类与缺量的代价完全不对称：

- 类存在但样本少：预测差、置信度低，自动落进 `review/`，循环能自愈；
- **类不存在（如 4 星英雄从未标注）：这些帧会被高置信度塞进最像的近邻类**（如 snow_star4 全进 snow_star3），且不触发 review，重训后错误自我强化，是循环里最危险的污染源。

因此启动循环前必须保证：**阵容中每个英雄的 star1~star4 在 train/labeled 中都有样本；单局凑不齐时，从其他局的 raw 或未标注簇中补捞**。宁可每类只有一二十张先跑起来，不可带着盲区类开跑。

### 人工情报比模型自纠错高效得多

翻 raw/候选图时发现「某局某格某帧段是某英雄某星级」，按 局号+帧区间+格子 精确定位，从 labeled（清污染）和 unclassified（捞增量）一次收齐。实测：dk_star3 靠 4 条人工情报从 5 张涨到 157 张，远快于等循环自己发现。

### 人工确认是抽样验收，不是逐张放行

每类随机抽 5~10 张判断该类整体质量即可；抽查发现的错组按真实类别改道（能认出真身的带干扰帧归真身类，认不出的进 unknown）。预期 2~3% 噪声可被按 类×局×类型 的均衡采样兜住；每轮重训在独立 validation 上复检，某类准确率突然下降即提示该类建议有系统性错误。

### 需要人工兜底的固定雷区

- 相邻星级（star1↔star2、star2↔star3）是模型长期弱点，重点抽查；
- 伤害跳字、技能特效会把英雄「染」成其他类（实测 3 星死骑被大量认成 angel_star3），这类帧反而是最值钱的困难样本，归真实类而非丢弃；
- 拖动合成时的过渡帧、完全遮挡帧没有真身，一律 `unknown`，不得猜测凑数。

### 每轮迭代的固定节奏

```text
搬运/sync-labels -> 重训 -> evaluate 到 validation
  -> 看混淆矩阵：某类大量流向近邻类 = 缺类或污染信号
  -> 用 seed 模型的 prediction_manifest 反查可疑帧（top1/top2 组合）定位污染
```

## 八、最终独立验证

```powershell
wlxq-bot hero-classifier evaluate $root `
  --model outputs/hero_classifier/assault-helper/hero_classifier.onnx `
  --split test `
  --output outputs/hero_classifier/assault-helper-test
```

默认读取 `$root/test/labeled`，输出逐图预测 CSV、总体准确率、拒绝率和混淆统计。test 局不得在 train 或 validation 出现；该约束在 import 时即检查。

## 原图、裁切图、模型和 Git

- `datasets/` 已被 `.gitignore` 排除，不上传真实截图和人工标签；
- 裁切并确认 import 完整后，原始 `rounds/<局>/raw/*.png` 技术上可删除以节省空间；
- 更建议至少保留到该批裁切、标注、训练和测试完成，以便坐标或标签问题时重裁；
- 裁切后的已标注英雄图应长期保留，后续可继续增加数据并重新训练；
- 模型二进制通常不直接放普通 Git：体积增长快、差异不可审查。需要共享时用 Git LFS、Release 或模型存储，并同时保留模型 metadata 与训练数据版本说明。

存储量取决于截图复杂度和 PNG 压缩。360 张完整 PNG 往往需要数百 MB 到数 GB；4320 张小格 PNG 通常还需数百 MB。训练本身额外需要 PyTorch 环境、缓存、checkpoint 和 ONNX，建议至少预留 10～20 GB 空间，正式采集前以一局实际目录大小外推。

## 分辨率说明

模型输入会统一缩放到固定尺寸，因此不是只能识别训练电脑的桌面分辨率。但识别对象来自裁切后的英雄格：另一台电脑必须使用正确的窗口客户区比例和格子坐标，且游戏 UI 缩放、画质、特效风格不能与训练数据差异过大。

最稳妥的做法：

1. 每种实际运行环境先重新确认 12 格裁切预览；
2. 用目标电脑采少量测试局加入独立 test；
3. 若跨分辨率效果下降，补充该环境的 train 数据后重新训练；
4. 不因模型支持 resize 就假定任意分辨率零验证可用。

## 常见问题

### import 提示对局已经登记

程序会提示该局原来所属的 split/import 并自动跳过，其余新局继续处理。同一局不能借再次请求从 train 改放 validation/test。若一批请求全部被跳过，则不会创建空 import。需要核对时检查三个 split 的 `imports/*/rounds.txt`；若确需重新划分，应先保留数据并明确迁移旧 import 和标签，不能简单复制同一局。

### 训练找不到图片

确认图片已经移动到 `<split>/labeled/`，文件名未修改，并执行过：

```powershell
wlxq-bot hero-classifier sync-labels $root --split train
wlxq-bot hero-classifier sync-labels $root --split validation
```

### 某些格子裁偏

立即停止该环境的数据制作，检查窗口客户区、DPI、角色和 `configs/tasks.yaml`。不要把裁偏图片继续标注，也不要猜坐标。
