# PSMF-MTP

[English](README.md) | 简体中文

PSMF-MTP 用于多功能治疗肽预测，结合预训练序列编码器、八组理化及结构特征、Mamba 融合与 Transformer 分类器。

本指南介绍项目整体流程、已提供的模型权重及数据、自有数据准备方式，以及 PyCharm 本地运行演示视频。

## 公开资源与实现范围

| 资源 | 位置 |
|---|---|
| 基准序列及 21 标签注释 | `dataset/` |
| 八组特征及合并后的 3,387 维矩阵 | `features/` |
| 冻结的短肽序列编码器权重及分词器 | `pre_trained_model/` |
| 训练100轮的融合与分类部分权重 | `weights/psmf_mtp_100epochs_seed205808.pt` |
| 检查点设置及单次训练记录 | `weights/model_config.json`、`weights/training_record.json` |
| 序列、标签和特征对应的小型输入示例 | `examples/small_demo/` |
| 数据准备、序列表示提取、权重检查和评估工具 | `scripts/` |
| PyCharm/WSL 本地运行演示 | [观看视频](videos/brief%20introduce.mp4) |

现阶段，核心模型结构实现、自定义损失实现及完整训练/推理入口保留在本地。公开工具可以准备输入、校验权重及评估已保存的预测结果。视频中的完整预测使用包含这些核心实现的本地项目；单独下载参数字典并不包含相应模型实现。

## 项目整体流程

1. 准备肽序列及功能注释，保持训练集与测试集独立。
2. 使用冻结的 UniRef90 短肽编码器获得序列表示。
3. 准备 AAC、DDE、CKSAAGP、PAAC、APAAC、AAindex1、CTD 和 MorganFP 八组特征，并按固定顺序合并。
4. 将序列表示和描述符映射到共同表示空间，通过 Mamba 进行融合。
5. 使用 Transformer 分类器和 sigmoid 输出得到 21 个功能标签的概率。
6. 训练时保持序列编码器冻结，优化映射、融合和分类部分，并保存训练好的权重。
7. 推理时加载对应编码器与预测模型权重，按配置阈值获得预测标签集合。
8. 已知真实标签时，计算评价指标并检查逐标签表现。

论文还分析了方法比较、特征及组件消融、描述符维度、编码器下游迁移、辅助标签上下文、逐标签性能，以及表示和特征组解释。

## 环境配置与下载

公开工具要求 Python 3.10 或更高版本。大型文件通过 Git LFS 下载：

```
git lfs install
git clone https://github.com/MuJianhang/PSMF-MTP.git
cd PSMF-MTP
git lfs pull
python -m venv .venv
```

在 Windows PowerShell 中激活环境：

```
.\.venv\Scripts\Activate.ps1
```

在 Linux 或 macOS 中激活环境：

```
source .venv/bin/activate
```

安装公开工具所需依赖：

```
python -m pip install -r requirements.txt
```

预处理和权重检查工具可以在 CPU 上运行。完整本地模型还需要保留的核心实现及对应 Mamba 依赖；论文使用 Ubuntu、WSL 和 CUDA 环境。在 PyCharm 中应选择实际安装了这些依赖的解释器。

## 已提供的训练权重

该检查点包含序列映射、八组描述符映射、Mamba 融合、Transformer 分类器及输出层，共6,877,163个参数值。冻结编码器另存于 `pre_trained_model/`；完整预测需要两部分共同使用。

| 设置 | 所选检查点 |
|---|---|
| 特征配置 | 完整的3,387维描述符 |
| 训练轮数 / 随机种子 | 100 / 205808 |
| 批大小 / 优化器 | 64 / AdamW |
| 学习率 / 权重衰减 | `8e-5` / `0.15` |
| Dropout | `0.4` |
| 融合维度 / Token 数 | 80 / 41 |
| Mamba | 1个块；状态16；卷积4；扩展4 |
| Transformer | 1层；4个头；前馈维度256 |
| 输出标签数 / 阈值 | 21 / `0.57` |

这是100轮完整特征配置中的一次独立训练检查点。训练记录描述该次运行，其指标不是五次运行的均值。权重以 PyTorch `state_dict` 提供，预测时需要匹配的模型定义。配置与 SHA-256 校验值记录在 [weights/model_config.json](weights/model_config.json)。

检查公开文件并查看真实权重：

```
python scripts/verify_release.py
python scripts/inspect_model_weights.py
```

成功检查后分别显示 `PUBLIC RELEASE CHECK: PASS` 和 `PSMF-MTP WEIGHTS CHECK: PASS`。

## 示例输入

小型示例包含128条训练记录和32条测试记录，以及对应描述符矩阵：

```
examples/small_demo/
|-- dataset/
|   |-- train_new.fasta
|   `-- test_new.fasta
|-- features/merged/
|   |-- train_features_3387.npy
|   `-- test_features_3387.npy
`-- provenance.json
```

解析注释并导出标签矩阵：

```
python scripts/read_fasta_labels.py --dataset_dir examples/small_demo/dataset --out_dir outputs/small_demo/labels --save_npy
```

程序保存序列/标签 CSV，以及 `train_labels.npy` 和 `test_labels.npy`。获取与所选检查点对应的编码器输入：

```
python scripts/extract_sequence_features.py --fasta examples/small_demo/dataset/test_new.fasta --output outputs/small_demo/test_sequence_features.npy --device cpu
```

序列表示矩阵形状为 `(32, 6400)`；示例测试描述符矩阵为 `(32, 3387)`。两者均保持原始 FASTA 记录顺序。

## 使用自己的数据集

### 1. 准备序列与标签

使用相同的21个功能定义及标签顺序。序列提取和预测可以使用普通 FASTA 标识符；训练或评估需要已知标签。带标签的头部可以包含样本标识符及21位二进制值：

```
>peptide_001 labels=010000100000000000000
ARRRRCSDRFRNCPADEALCGRRRR
```

该记录用于说明格式，实际标签应来自自己的数据注释。第2位和第7位分别对应 ABP 和 AFP。

将自己的文件与公开基准分开保存：

```
data/my_dataset/
|-- dataset/
|   |-- train_new.fasta
|   `-- test_new.fasta
`-- features/merged/
    |-- train_features_3387.npy
    `-- test_features_3387.npy
```

```
python scripts/read_fasta_labels.py --dataset_dir data/my_dataset/dataset --out_dir outputs/my_dataset/labels --save_npy
```

检查点固定输出21个标签。如果目标标签定义不同，需要相应调整模型输出并重新训练。

### 2. 准备序列表示

```
python scripts/extract_sequence_features.py --fasta data/my_dataset/dataset/test_new.fasta --output outputs/my_dataset/test_sequence_features.npy --device cpu
```

准备训练输入时，将 FASTA 改为训练文件，并使用另一个输出路径。无标签 FASTA 也可以使用相同命令处理。可以通过 `--max_samples` 先处理少量记录。

为匹配所选100轮检查点，工具使用50个 Token 位置，其中包含 `[CLS]` 和 `[SEP]`，对长度为1–50个残基的输入最多保留前48个残基。它将全部50个位置的隐藏表示展平为6,400个值，并先按训练缓存方式舍入为 float16，再导出 float32 输入。配套的 `.ids.csv` 与 `.metadata.json` 记录样本顺序及预处理细节。

### 3. 准备描述符特征

每个划分都需要形状为 `(number_of_samples, 3387)`、所有值均为有限值的矩阵，行顺序与序列完全一致。特征定义和参数采用论文中的设置。

如果八组特征文件已经准备好，可使用以下命令合并：

```
python scripts/load_features_demo.py --feature_root data/my_dataset/features --out_dir data/my_dataset/features/merged
```

| 特征组 | 训练文件名 | 存储列数 | 模型输入列数 |
|---|---|---:|---:|
| AAC | `AAC/train_aac.csv` | 20 | 400 |
| DDE | `DDE/train_dde.csv` | 400 | 400 |
| CKSAAGP | `CKSAAGP/train_cksaagp.csv` | 100 | 400 |
| PAAC | `PAAC/train_paac.csv` | 22 | 396 |
| APAAC | `APAAC/train_apaac.csv` | 24 | 384 |
| AAindex1 | `AAindex1/train_aaindex1.npy` | 566 | 566 |
| CTD | `CTD/train_ctd.npy` | 441 | 441 |
| MorganFP | `MorganFP/train_morganfp_400.npy` | 400 | 400 |

测试文件使用 `test` 前缀。CSV 特征文件不含表头或索引列。特征按表格顺序拼接。加载工具负责合并已有特征组，公开工具中尚未包含从任意原始 FASTA 完整计算八组特征的全部实现。

使用公开检查点推理时，保持原特征定义与数值尺度，并复用 `features/MorganFP/` 中在训练阶段拟合的 Morgan 预处理对象。该检查点不额外对完整描述符矩阵做全局标准化。如果重新训练模型，涉及数据拟合的预处理只使用训练数据。

### 4. 在完整本地项目中运行模型

在 PyCharm 中打开完整 PSMF-MTP 本地项目，选择已经配置好的解释器，并将工作目录设为该项目根目录。在本地保留的推理入口中，指定输入 FASTA、顺序对应的6,400维序列表示和3,387维描述符、预训练编码器位置、公开预测权重、阈值 `0.57` 及输出目录。将权重加载到匹配的模型结构，切换至评估模式后运行预测。

核心入口保留在本地，本次未作为公开脚本提供。[演示视频](videos/brief%20introduce.mp4)展示了实际运行配置和执行过程。更换兼容数据集时，需要同时替换序列/注释路径及对应特征路径；仅加载形状相同但来源不同的特征矩阵无法保证正确对应。

## 结果输出与评估

本地推理阶段应导出形状为 `(number_of_samples, 21)` 的概率矩阵、阈值化预测及样本/序列对应信息，评估时保持相同行顺序。视频中可以通过 CSV 表格查看序列标识符、预测标签及概率。

对于保存的得分及对应真实标签，将下方示例预测路径替换为实际本地输出：

```
python scripts/evaluate_predictions.py --true outputs/small_demo/labels/test_labels.npy --pred path/to/predicted_scores.npy --threshold 0.57 --output outputs/small_demo/metrics.json
```

评估工具支持 `.npy` 或不含表头的数值 CSV 矩阵，输出 Accuracy（样本级 Jaccard 相似度）、Precision、Coverage（样本级召回率）、Absolute true（标签集合完全匹配比例）及 Absolute false（Hamming loss）。计算指标需要真实标签；无标签肽仍可进行预测。二进制预测矩阵直接计算指标。

固定标签顺序如下：

```
AAP, ABP, ACP, ACVP, ADP, AEP, AFP, AHIVP, AHP, AIP, AMRSAP,
APP, ATP, AVP, BBP, BIP, CPP, DPPIP, QSP, SBP, THP
```

另见 [dataset/LABELS.md](dataset/LABELS.md)。完整基准包含7,872条训练肽和1,969条测试肽；小型示例的样本来源索引记录在 `provenance.json`。

## PyCharm 本地运行视频

[观看 PSMF-MTP 操作演示视频](videos/brief%20introduce.mp4)。

视频展示了示例输入、使用公开的100轮训练权重进行预测、预测标签输出、数据集路径位置，以及在 PyCharm 中通过 WSL 和 CUDA 进行3轮训练预览。训练预览用于展示程序运行过程，其指标不代表最终模型性能。

录屏使用本地30条测试记录的子集，公开的小型示例仍包含32条测试记录。视频展示完整本地项目的操作流程，与上述数据准备及评估说明配合使用。

## 引用与联系

使用这些资源时，请引用对应的 PSMF-MTP 论文。如有问题，可通过 [GitHub issues](https://github.com/MuJianhang/PSMF-MTP/issues) 联系。
