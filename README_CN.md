# M2KT：基于多模态知识迁移的放射学报告自动生成

从胸部 X 光片自动生成放射学报告的深度学习框架，包含自学习知识库与多模态对齐机制，并在此基础上拓展了五个前沿研究方向。

---

## 项目概述

本项目研究从胸部 X 光片自动生成放射学描述报告。基线架构采用编码器-解码器范式，包含两个核心机制：

1. **自学习知识库**：维护一个可学习的记忆矩阵，训练时通过注意力机制从真实报告文本中更新知识，推理时利用预测的临床标签从知识库中检索先验知识，辅助解码器生成更准确的报告。
2. **多模态对齐**：基于 Hinge 的对比排序损失，使用标签感知的动态边距将图像特征与对应的报告文本特征在嵌入空间中对齐，拉近正样本对、推远负样本对。

```
X光图像 → 视觉提取器(CNN) → 图像特征 + 标签预测
                                    ↓
                              知识库检索（标签嵌入查询）
                                    ↓
                         Transformer编解码器 → 放射学报告
                                    ↓
               语言模型损失 + 标签损失 + 排序损失 → 联合训练
```

---

## 五个研究方向

所有新增代码位于 `innovations/` 目录，每个方向独立、自包含，复用共享的 `models/` 和 `modules/` 基础模块。

### 方向一：MLLM 增强的可进化知识库

**目录**：`innovations/mllm_knowledge_base/` | **入口**：`python -m innovations.mllm_knowledge_base.main_mllm`

将标准 CNN 视觉编码器替换为预训练的医学视觉-语言模型（BiomedCLIP、BioViL、CheXzero），并引入三层分层记忆结构（实例级 → 疾病级 → 跨疾病关联记忆槽），配合 LSTM 风格的门控更新机制，防止知识积累过程中的灾难性遗忘。

| 模块 | 说明 |
|------|------|
| 医学 VLM 编码器 | BiomedCLIP / BioViL / CheXzero 作为视觉主干 |
| 分层记忆库 | 三层结构：实例级、疾病级、跨疾病关联 |
| 门控记忆更新 | 可学习的遗忘/写入门控，实现可控知识进化 |
| 自适应控制器 | 基于新颖性检测动态调节每样本记忆更新强度 |

### 方向二：基于 Agent 的迭代自省式报告生成

**目录**：`innovations/agent_refinement/` | **入口**：`python -m innovations.agent_refinement.main_agent`

模拟放射科医生"首读→复查→修正"的工作流程：基础生成器产出初稿，Refiner Agent 利用视觉反馈迭代修正。一致性检查器检测报告内部矛盾，完整性验证器发现遗漏的病灶描述，通过直接偏好优化（DPO）将精炼过程与临床质量偏好对齐。

| 模块 | 说明 |
|------|------|
| 草稿→精炼流水线 | 基于交叉注意力的迭代报告改进 |
| 一致性检查器 | NLI 风格双向蕴涵检测 + 临床规则库 |
| 完整性验证器 | CheXpert 标签与报告内容交叉比对 |
| DPO 训练 | 直接偏好优化对齐精炼行为与临床质量 |

### 方向三：图文证据对齐与可解释生成

**目录**：`innovations/evidence_grounding/` | **入口**：`python -m innovations.evidence_grounding.main_grounding`

从解码器交叉注意力中提取空间热力图，将生成的每一句报告链接到图像的具体区域，打破"黑盒"生成。包含基于 Monte Carlo Dropout 的不确定性量化方法，分离偶然不确定性与认知不确定性，输出每个报告的可信度评分。

| 模块 | 说明 |
|------|------|
| 空间注意力头 | 逐 Token 和句子级空间热力图提取 |
| 证据对齐损失 | BBox 弱监督 + 对比接地 + 平滑正则化 |
| 不确定性估计器 | MC Dropout 分离偶然/认知不确定性 |
| 可靠性评分 | 学习的临床可信度评估 |

### 方向四：图神经网络 + 医学本体知识增强

**目录**：`innovations/gnn_ontology/` | **入口**：`python -m innovations.gnn_ontology.main_gnn`

构建包含层级、共病、因果、解剖关系的胸部 X 光异质知识图谱。2 层图注意力网络（GAT）在疾病节点间传播结构化知识，可学习门控融合机制自适应平衡 GNN 推理路径与数据驱动的记忆检索路径。

| 模块 | 说明 |
|------|------|
| 胸部 X 光知识图谱 | 层级 + 共病 + 因果 + 解剖关系边 |
| GAT 推理器 | 2 层图注意力网络跨疾病知识传播 |
| 门控融合 | 上下文感知的 GNN 知识与记忆检索融合 |
| 内置本体库 | 14 个 CheXpert 标签 + 临床关系，可扩展 UMLS/RadLex |

### 方向五：时序对比学习与疾病进展建模

**目录**：`innovations/temporal_progression/` | **入口**：`python -m innovations.temporal_progression.main_temporal`

处理同一患者的纵向配对胸部 X 光片，建模疾病随时间演变。Siamese 视觉编码器配合跨时间注意力和差值编码，预测每个病灶的进展状态（稳定/好转/恶化）。时序一致性损失强制严重程度平滑轨迹，对齐报告文本差异与视觉变化。

| 模块 | 说明 |
|------|------|
| 纵向配对数据 | MIMIC-CXR 同患者多次检查，可配置时间间隔 |
| Siamese 时序编码器 | 共享 CNN + 跨时间注意力 + 差值编码 |
| 进展分类器 | 每病灶三分类：稳定 / 好转 / 恶化 |
| 时序一致性损失 | 严重程度平滑 + 报告-文本对齐 + 进展监督 |

---

## 项目结构

```
├── config/                          # YAML 配置文件和参数解析
│   ├── iu.yml, mimic.yml            # 数据集基础配置
│   ├── iu_resnet.yml                # IU + ResNet-101
│   ├── iu_densenet.yml              # IU + DenseNet-121
│   ├── iu_efficientnet.yml          # IU + EfficientNet-b0
│   ├── mimic_resnet.yml             # MIMIC + ResNet-101
│   ├── config.py                    # 支持 _BASE_ 继承的配置加载器
│   └── opts.py                      # 命令行参数解析
│
├── models/                          # 模型定义
│   ├── lamrg.py                     # 模型变体 v7~v12, v91, v92, vRebuttal
│   ├── baseline.py                  # 纯 Transformer 基线
│   └── LabelFeature.py              # 标签特征注入变体
│
├── modules/                         # 核心模块
│   ├── visual_extractor.py          # 视觉特征提取器（ResNet/DenseNet/EfficientNet）
│   ├── text_encoder.py              # Transformer 文本编码器 + MHA_FF
│   ├── Transformer.py               # 完整 Transformer（编码器、解码器、注意力）
│   ├── AttModel.py                  # 注意力模型基类（Beam Search）
│   ├── CaptionModel.py              # 图像描述生成基类（序列解码策略）
│   ├── tokenizers.py                # 分词器（词表构建、报告清洗）
│   ├── datasets.py                  # IU（多图）/ MIMIC（单图）数据集
│   ├── dataloaders.py               # 数据加载器（增强、批处理）
│   ├── loss.py                      # 语言模型损失 + 标签 BCE 损失 + 排序损失
│   ├── trainer.py                   # 训练器（训练循环、验证、检查点、早停）
│   ├── metrics.py                   # 评价指标（BLEU 1~4、ROUGE-L、CIDEr、METEOR）
│   ├── optimizers.py                # 优化器和学习率调度器
│   ├── utils.py                     # 解码辅助工具
│   └── efficientnet_pytorch/        # 内置 EfficientNet 实现
│
├── misc/
│   └── utils.py                     # 日志设置与图像归一化
│
├── innovations/                     # 五个研究方向
│   ├── mllm_knowledge_base/         # [1] MLLM 增强知识库
│   ├── agent_refinement/            # [2] Agent 迭代自省
│   ├── evidence_grounding/          # [3] 证据接地与可解释性
│   ├── gnn_ontology/                # [4] GNN + 本体增强
│   └── temporal_progression/        # [5] 时序疾病进展建模
│
├── main.py                          # 入口：完整模型（含知识库）
├── main_basic.py                    # 入口：基础 Transformer 基线
├── main_label_feature.py            # 入口：标签特征注入
├── mainbaseline.py                  # 入口：最小基线
├── README.md                        # 英文文档
└── README_CN.md                     # 中文文档（本文件）
```

---

## 环境要求

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | >= 3.8 | 运行环境 |
| PyTorch | >= 1.7 | 深度学习框架 |
| torchvision | — | 预训练 CNN 模型 |
| tensorboardX | — | 训练可视化 |
| tqdm | — | 进度条 |
| numpy | — | 数值计算 |
| pandas | — | 数据处理 |
| pyyaml / yacs | — | 配置管理 |
| Pillow | — | 图像加载 |
| pycocoevalcap | — | MS COCO 评价指标工具包 |
| CheXpert-labeler | — | 外部标签提取工具 |
| transformers | 可选（方向一） | BiomedCLIP/BioViL 集成 |

## 数据准备

下载数据集并放入 `data/` 目录：

- **IU X-Ray**：[印第安纳大学胸部 X 光数据集](https://iuhealth.org/find-medical-services/x-rays) — 约 7,470 张图像，每份报告对应正位+侧位两个视角
- **MIMIC-CXR**：[PhysioNet MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) — 约 377,110 张图像，每份报告对应单张图像

下载评估工具包：

```bash
git clone https://github.com/tylin/coco-caption.git pycocoevalcap/
```

---

## 训练命令

### 基线模型

```bash
# 完整模型（含知识库 + 多模态对齐）
python main.py --cfg config/iu_resnet.yml --expe_name 实验名称 \
               --label_loss --rank_loss --version 12

# 基础 Transformer 基线
python main_basic.py --cfg config/iu_resnet.yml --expe_name 实验名称 \
                     --label_loss --rank_loss --version 91

# 消融：无自学习知识库
python main.py --cfg config/iu_resnet.yml --expe_name 实验名称 \
               --label_loss --rank_loss --version 92

# 消融：无多模态对齐
python main.py --cfg config/iu_resnet.yml --expe_name 实验名称 \
               --version 12     # 去掉 --rank_loss
```

### 五个研究方向

```bash
# 方向一：MLLM 增强知识库
python -m innovations.mllm_knowledge_base.main_mllm \
    --cfg config/iu_resnet.yml --expe_name mllm_v1 \
    --vlm_type biomedclip --freeze_vlm --label_loss --rank_loss

# 方向二：Agent 迭代自省
python -m innovations.agent_refinement.main_agent \
    --cfg config/iu_resnet.yml --expe_name agent_v1 \
    --max_refine_steps 3 --use_dpo

# 方向三：证据接地
python -m innovations.evidence_grounding.main_grounding \
    --cfg config/iu_resnet.yml --expe_name grounding_v1 \
    --label_loss --rank_loss --grounding_weight 0.1

# 方向四：GNN + 本体增强
python -m innovations.gnn_ontology.main_gnn \
    --cfg config/iu_resnet.yml --expe_name gnn_v1 \
    --ontology_source builtin --label_loss --rank_loss

# 方向五：时序进展建模
python -m innovations.temporal_progression.main_temporal \
    --cfg config/mimic_resnet.yml --expe_name temporal_v1 \
    --max_time_delta 365 --label_loss --rank_loss
```

---

## 模型版本速查表

| 版本号 | 类名 | 说明 |
|--------|------|------|
| 900 | `baseline.py` | 纯 Transformer 编解码基线 |
| 7 | `LAMRGModel_v7` | 简单标签特征拼接 |
| 8 | `LAMRGModel_v8` | 引入 TextEncoder + prior_memory + select_prior |
| 9 | `LAMRGModel_v9` | 线性投影层；返回 (z_img, z_txt) 用于排序损失 |
| 10 | `LAMRGModel_v10` | 通过 `linear_mem` 投影记忆 |
| 11 | `LAMRGModel_v11` | Mask + query_mem + linear_z + update_memory |
| **12** | **`LAMRGModel_v12`** | **完整版**：标签嵌入查询 + 独立 memory heads |
| 91 | `LAMRGModel_v91` | 消融：仅标签损失，无记忆 |
| 92 | `LAMRGModel_v92` | 消融：标签损失 + 排序损失，无记忆 |

---

## 评估指标

| 指标 | 说明 |
|------|------|
| BLEU-1 ~ BLEU-4 | 生成报告与真实报告的 n-gram 重叠度 |
| ROUGE-L | 最长公共子序列召回率 |
| CIDEr | 基于 TF-IDF 加权的共识图像描述评估 |
| METEOR | 基于词对齐和同义词匹配的评估 |

所有指标通过 `pycocoevalcap` 工具包计算。

---

## 主要参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--cfg` | str | 必填 | YAML 配置文件路径 |
| `--expe_name` | str | 必填 | 实验名称（用于日志和保存） |
| `--version` | int | 12 | 模型版本号 |
| `--label_loss` | flag | False | 启用标签预测 BCE 损失 |
| `--rank_loss` | flag | False | 启用多模态排序损失 |
| `--batch_size` | int | 16 | 训练批次大小 |
| `--epochs` | int | 50 | 训练轮数 |
| `--lr_ed` | float | 1e-4 | 编解码器学习率 |
| `--lr_ve` | float | 1e-5 | 视觉提取器学习率 |
| `--grad_clip` | float | 5.0 | 梯度裁剪阈值 |
| `--early_stop` | int | 10 | 早停等待轮数 |
| `--seed` | int | 42 | 随机种子 |
