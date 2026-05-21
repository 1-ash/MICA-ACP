# MICA-ACP 实验设计方案

本文档用于规划 MICA-ACP 的论文实验部分。设计思路参考用户提供的 PKDF-Net 论文实验组织方式，但实验内容按当前 MICA-ACP 代码、数据文件和已有结果重新整理。

核心目标是：不仅报告一个最优结果，还要系统证明 MICA-ACP 在不同数据集、不同任务场景、不同模块组合和不同超参数下的有效性、稳健性与可解释性。

## 2. 参考论文实验结构可借鉴点
参考论文的实验设置大致包含以下模块，MICA-ACP 应尽量对齐这种完整度：
1. 多个 benchmark 数据集上的主结果。
2. 与已有 SOTA 方法进行系统对比。
3. 使用 ACC、Precision、Sensitivity、Specificity、F1、AUC、MCC 等多指标评价。
4. 对核心模块做消融实验。
5. 对序列长度和截断策略做敏感性分析。
6. 对关键模块数量和超参数做敏感性分析。
7. 做模块替换实验，证明当前模块选择优于常见替代结构。
8. 做统计显著性检验。
9. 分析推理时间和计算成本。
10. 使用 ROC、PR、t-SNE、融合权重或注意力热图做可视化解释。

MICA-ACP 的实验可以按这个框架展开，但需要结合本模型特有模块：ESM-C 表征、BLOSUM、AAindex、AAC20、mRMR 筛选、P0 与 ESM 的 cross-attention、多尺度 CNN、GatedAddFusion、post-fusion BiGRU、EMA、R-Drop 和 hard negative mining。

## 3. 数据集实验设计
当前项目中已经包含参考论文使用的五类 benchmark 数据集，可以直接作为主实验数据。
| 简称 | 训练文件 | 测试文件 | Train 正/负 | Test 正/负 | 长度范围 |
|---|---|---|---:|---:|---|
| Main | `ACP2_main_train.tsv` | `ACP2_main_test.tsv` | 689 / 689 | 172 / 172 | 2-50 |
| Alternate | `ACP2_alternate_train.tsv` | `ACP2_alternate_test.tsv` | 776 / 776 | 194 / 194 | 3-50 |
| FL | `ACP_FL_train_500.tsv` | `ACP_FL_test_164.tsv` | 250 / 250 | 82 / 82 | 11-207 |
| Fuse | `ACPred-Fuse_ACP_Train500.tsv` | `ACPred-Fuse_ACP_Test2710.tsv` | 250 / 250 | 82 / 2628 | 7-207 |
| Mixed80 | `ACP-Mixed-80-train.tsv` | `ACP-Mixed-80-test.tsv` | 242 / 242 | 61 / 61 | 11-207 |
建议将 Main 和 Alternate 作为主数据集，用于主要对比、消融和可视化；FL、Fuse、Lee、Mixed80 作为泛化性和稳健性验证数据集。

### 3.1 必做数据集实验
| 实验编号 | 数据集 | 目的 | 是否必做 |
|---|---|---|---|
| D1 | Main | 主任务性能评估 | 必做 |
| D2 | Alternate | 另一 ACP2.0 场景下验证泛化能力 | 必做 |
| D3 | FL | 小规模平衡测试集上的鲁棒性 | 必做 |
| D4 | Fuse | 极度类别不平衡测试集上的鲁棒性 | 必做 |
| D6 | Mixed80 | 低相似性冗余控制数据集验证 | 必做 |

## 4. 评价指标与报告方式
所有实验统一报告以下指标：
| 指标 | 说明 |
| ACC | 总体准确率 |
| Precision / Prec | 预测为 ACP 的样本中真正 ACP 的比例 |
| Sensitivity / SE / Recall | 正样本识别率 |
| Specificity / SP | 负样本识别率 |
| F1 | Precision 与 Recall 的调和平均 |
| AUC | ROC 曲线下面积 |
| MCC | 对二分类最稳健的综合指标，尤其适合不平衡数据 |


### 4.1 阈值设置
每个实验建议同时报告两种阈值结果：
1. 固定阈值 `0.5`。
2. 在验证集上按 MCC 搜索得到的最佳阈值。
论文主表使用验证集阈值结果。

## 5. 主结果实验
### 5.1 推荐命令模板
以 Main 数据集为例：
```powershell
conda run -n ESM python train.py `
  --train_tsv data/ACP_dataset/ACP2_main_train.tsv `
  --test_tsv data/ACP_dataset/ACP2_main_test.tsv `
  --select_aaindex_mrmr `
  --aaindex_dim 60 `
  --lr 5e-5 `
  --batch_size 64 `
  --seed 42 `
  --run_name strict_main_k60_mRMR_lr5e-5_seed42 `
  --no-use_test_as_valid `
  --no-use_swanlab
```

Alternate 数据集：
```powershell
conda run -n ESM python train.py `
  --train_tsv data/ACP_dataset/ACP2_alternate_train.tsv `
  --test_tsv data/ACP_dataset/ACP2_alternate_test.tsv `
  --select_aaindex_mrmr `
  --aaindex_dim 60 `
  --lr 5e-5 `
  --batch_size 64 `
  --seed 42 `
  --run_name strict_alternate_k60_mRMR_lr5e-5_seed42 `
  --no-use_test_as_valid `
  --no-use_swanlab
```
其余数据集只需要替换 `--train_tsv`、`--test_tsv` 和 `--run_name`。

### 5.2 主结果表设计
建议将六个数据集分成三张表，写法接近参考论文：
| 表编号 | 内容 |
|---|---|
| Table 1 | 六个数据集的训练/测试样本统计 | ✅️
| Table 2 | Main 和 Alternate 上的 SOTA 对比 |
| Table 3 | FL 和 Fuse 上的泛化对比 |
| Table 4 | Mixed80 上的独立/低冗余测试结果 |

每张性能表统一列：
```text
Dataset | Method | ACC | Precision | SE | SP | F1 | AUC | MCC
```

### 5.3 已有日志结果（Main）
已核对 `logs/main_k60_mRMR_lr5e-5_seed42_test*` 与对应 `checkpoints/*/best.pt` 中保存的真实配置。目前只有 Main 数据集结果，且所有 run 均使用 `ACP2_main_train.tsv` / `ACP2_main_test.tsv`、`use_test_as_valid=True`、`Lmax=50`、`aaindex_dim=60`、`select_aaindex_mrmr=True`、`batch_size=64`、`lr=5e-5`、`seed=42`。下表按 4.1 的论文主表口径，采用 `test_metrics_at_best_threshold`。

| Dataset | Method / run | 关键设置 | Best epoch | Threshold | ACC | Precision | SE | SP | F1 | AUC | MCC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Main | Full MICA-ACP / `main_k60_mRMR_lr5e-5_seed42_test2` | `w_hn_max=1.2`, `lambda_hn=1`, `h_embed_update=every_n_epochs`, `h_embed_ema_decay=0.95` | 74 | 0.5775 | 0.8140 | 0.8140 | 0.8140 | 0.8140 | 0.8140 | 0.8751 | 0.6279 |

## 6. SOTA 对比实验

### 6.1 传统机器学习或手工特征方法

引用原论文结果：
```text
AntiCP 2.0
ACPred-LAF
ACP-check
```


### 6.2 深度学习方法
引用原论文结果
```text
ACP-DL
CACPP
con_ACP
ACP-CapsPred
```

### 6.3 预训练模型或近期强基线
引用论文结果：
```text
UniDL4BioPep
PKDF-Net
```

## 7. 消融实验设计
消融实验建议主要在 Main 和 Alternate 上完成，必要时在 FL 或 Fuse 上补充验证。
### 7.1 特征来源消融
| 实验编号 | 变体 | 目的 | 当前代码支持情况 |
|---|---|---|---|
| F0 | Full MICA-ACP | 完整模型 | 已支持 |
| F1 | 去掉 AAindex | 验证 AAindex 贡献 | `--aaindex_dim 0` |
| F2 | 去掉 AAC20 | 验证全局组成先验贡献 | `--no_aac20_prior` |
| F3 | 去掉 mRMR，使用默认 Top60 | 验证 mRMR 筛选贡献 | 不传 `--select_aaindex_mrmr`，保持 `--aaindex_dim 60` |
| F4 | 使用全部 AAindex 553 维 | 验证高维 AAindex 是否带来噪声 | `--aaindex_dim 553` |
| F5 | 去掉 BLOSUM | 验证 BLOSUM 贡献 | 需要新增开关 |
| F6 | 仅 ESM-C | 验证预训练语义特征单独效果 | 需要新增模型变体 |
| F7 | 仅 P0 | 验证理化先验单独效果 | 需要新增模型变体 |
| F8 | 仅 token-CNN | 验证局部序列模式单独效果 | 需要新增模型变体 |

建议论文主消融表至少包含：
```text
Full
w/o AAindex
w/o AAC20
w/o mRMR
w/o BLOSUM
ESM-only
P0-only
token-CNN-only
```

### 7.2 AAindex 维度敏感性实验

用于回答：为什么选择 `k=60`。

建议设置：

```text
aaindex_dim = 0, 10,16, 20, 30, 32, 40, 48, 50, 60, 72, 100, 553
```

推荐命令示例：
```powershell
conda run -n ESM python train.py `
  --train_tsv data/ACP_dataset/ACP2_main_train.tsv `
  --test_tsv data/ACP_dataset/ACP2_main_test.tsv `
  --select_aaindex_mrmr `
  --aaindex_dim 48 `
  --lr 5e-5 `
  --seed 42 `
  --run_name main_aaidx48_mRMR_seed42 `
  --no-use_test_as_valid `
  --no-use_swanlab
```

结果建议画折线图：
```text
x-axis: AAindex dimension
y-axis: MCC / AUC / ACC
```

### 7.3 结构模块消融
| 实验编号 | 变体 | 目的 | 当前代码支持情况 |
|---|---|---|---|
| M0 | Full MICA-ACP | 完整结构 | 已支持 |
| M1 | 去掉 post-fusion BiGRU | 验证 BiGRU 贡献 | `--no-use_post_fusion_bigru` |
| M2 | cross-attention 方向改为 semantic_to_prior | 验证注意力方向 | `--cross_attention_direction semantic_to_prior` |
| M3 | 去掉 cross-attention | 验证 P0-ESM 交互贡献 | 需要新增开关 |
| M4 | 去掉 GatedAddFusion，改为直接相加 | 验证门控融合贡献 | 需要新增变体 |
| M5 | GatedAddFusion 改为 concat + MLP | 验证融合方式 | 需要新增变体 |
| M6 | GatedAddFusion 改为 SwiGLU fusion | 验证非线性融合 | 代码已有 `SwiGLUFusion`，但模型未接入 |
| M7 | BiGRU 替换为 LSTM/GRU/Transformer/Temporal Attention | 验证序列建模模块选择 | 需要新增变体 |

论文主消融表建议列：
```text
Full
w/o Cross-Attention
semantic_to_prior
w/o Gated Fusion
w/o BiGRU
Concat Fusion
SwiGLU Fusion
```

### 7.4 多尺度 CNN 消融
当前代码支持直接修改：
```text
--cnn_kernels
--num_cnn_layers
```

建议实验：
| 实验 | 参数 |
|---|---|
| 单尺度 | `[3]` |
| 双尺度 | `[3,5]` |
| 默认三尺度 | `[3,5,7]` |
| 大感受野三尺度 | `[5,7,9]` |
| 层数敏感性 | `num_cnn_layers = 0, 1, 2, 3, 4` |

注意：如果 `num_cnn_layers=0` 当前代码是否能完全跳过 CNN block 需要测试；若有问题，可新增显式 `--disable_cnn` 开关。

### 7.5 正则化与训练策略消融
Hard Negative 完整的实验应比较：
```text
w_hn_max = 1.0, 1.2, 1.5
```

#### 已有结果（Main，seed=42）
当前 `logs` 中已有 Hard Negative 相关消融结果，但没有发现 `w_hn_max=1.5` 的对应指标，因此下表只列已完成并可核对的 run。除表中设置外，其余公共配置为：Main 数据集、`use_test_as_valid=True`、`Lmax=50`、`aaindex_dim=60`、`select_aaindex_mrmr=True`、`AAC20=True`、`BLOSUM=True`、`ESM=True`、`P0=True`、`CNN kernels=[3,5,7]`、`num_cnn_layers=2`、`cross_attention_direction=prior_to_semantic`、`use_post_fusion_bigru=True`、`batch_size=64`、`lr=5e-5`。指标采用 `test_metrics_at_best_threshold`。

| 变体 | 对应 run | Hard Negative 设置 | Best epoch | Threshold | ACC | Precision | SE | SP | F1 | AUC | MCC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w/o Hard Negative | `test5` / `test8` | `lambda_hn=0.0`, `w_hn_max=1.0` | 96 | 0.6200 | 0.8081 | 0.8046 | 0.8140 | 0.8023 | 0.8092 | 0.8729 | 0.6163 |
| Hard Negative, H_embed 仅启动轮计算 | `test1` / `test9` | `lambda_hn=1.0`, `w_hn_max=1.1`, `h_embed_update=start_nhl_epoch_once` | 63 | 0.5485 | 0.8052 | 0.7966 | 0.8198 | 0.7907 | 0.8080 | 0.8713 | 0.6107 |
| Hard Negative, H_embed EMA 更新 | `test3` | `lambda_hn=1`, `w_hn_max=1.1`, `h_embed_update=every_n_epochs`, `h_embed_ema_decay=0.90` | 68 | 0.5037 | 0.8081 | 0.8011 | 0.8198 | 0.7965 | 0.8103 | 0.8761 | 0.6164 |
| Hard Negative, H_embed EMA 更新 | `test4` | `lambda_hn=1`, `w_hn_max=1.1`, `h_embed_update=every_n_epochs`, `h_embed_ema_decay=0.95` | 68 | 0.5037 | 0.8081 | 0.8011 | 0.8198 | 0.7965 | 0.8103 | 0.8761 | 0.6164 |
| Hard Negative, H_embed EMA 更新 | `test2` | `lambda_hn=1`, `w_hn_max=1.2`, `h_embed_update=every_n_epochs`, `h_embed_ema_decay=0.95` | 74 | 0.5775 | 0.8140 | 0.8140 | 0.8140 | 0.8140 | 0.8140 | 0.8751 | 0.6279 |

## 8. 超参数敏感性实验
这些实验不一定全部写入主文，但可以作为附录或补充材料。
| 参数 | 建议取值 |
|---|---|
| `lr` | `1e-5, 3e-5, 5e-5, 1e-4, 3e-4` |
| `batch_size` | `32, 64, 128` |
| `d` | `128, 256, 384, 512` |
| `num_heads` | `4, 8` |
| `dropout` | `0.1, 0.2, 0.3, 0.5` |
| `prior_dropout` | `0.1, 0.2, 0.3, 0.5` |
| `bigru_hidden` | `32, 64, 128` |
| `bigru_layers` | `1, 2` |

主文中建议只保留影响最大的 2-3 个超参数，例如：
```text
AAindex dimension
CNN kernel set
sequence length
```
其余放到附录。

## 9. 序列长度与截断策略实验
参考论文专门分析了序列长度和截断策略。MICA-ACP 也应该做，因为当前 `Lmax=50` 是模型和缓存的重要设计点。

### 9.1 最大长度敏感性
建议在 Main、Alternate、FL 上做：
```text
Lmax = 30, 40, 50, 60, 80, 100
```
Main 和 Alternate 大多不超过 50，因此主要验证 `50` 是否足够；FL、Fuse、Mixed80 存在长度超过 50 的样本，更适合验证截断策略。
注意：当前 `Lmax` 是配置字段，但命令行 parser 里没有显式 `--Lmax` 参数。建议新增：
```text
parser.add_argument("--Lmax", type=int)
```
并确保 ESM cache 文件名和 P0 构造随 `Lmax` 变化。


## 10. 统计显著性实验
参考论文使用 t-test 证明模型相对基线的提升显著。MICA-ACP 也建议做统计检验。
### 10.1 推荐做法
对以下指标做 paired t-test：
```text
ACC, F1, AUC, MCC
```

重点比较对象：
1. MICA-ACP vs 最强 SOTA baseline。
2. MICA-ACP vs ESM-only。
3. MICA-ACP vs w/o AAindex。
4. MICA-ACP vs w/o BiGRU。
5. MICA-ACP vs w/o Hard Negative。

显著性水平：
```text
alpha = 0.05
```

## 11. 可视化与解释性分析(这个图matplotlib能画吗)
建议至少做以下图：
| 图 | 内容 | 目的 |
|---|---|---|
| Fig. 1 | 模型结构图 | 展示 ESM/P0/CNN 三路输入和融合流程 |
| Fig. 2 | 六个数据集的序列长度分布 | 解释为什么选择 `Lmax=50` |
| Fig. 3 | ROC 和 PR 曲线 | 展示主模型与关键消融的分类能力 |
| Fig. 4 | 六数据集雷达图 | 直观比较 ACC/F1/AUC/MCC |
| Fig. 5 | pooled representation 的 t-SNE/UMAP | 展示融合表征的类间分离 |
| Fig. 6 | attention pooling 权重热图 | 分析模型关注的残基位置 |
| Fig. 7 | fusion gate 分布 | 解释 semantic/local 分支贡献 |
| Fig. 8 | hard negative 样本权重分布 | 说明 NHL 是否真的关注困难负样本 |
其中，完成：
```text
ROC/PR 曲线
t-SNE/UMAP
fusion gate 或 attention pooling 热图
```

## 12. 推荐表格与图片清单
### 12.1 表格
| 表编号 | 表名 |
|---|---|
| Table 1 | Benchmark 数据集统计 |
| Table 2 | MICA-ACP 超参数设置 |
| Table 3 | Main 和 Alternate 上的 SOTA 对比 |
| Table 4 | FL 和 Fuse 上的泛化结果 |
| Table 5 | Lee 和 Mixed80 上的独立测试结果 |
| Table 6 | 统计显著性检验 p-value |
| Table 7 | 特征来源消融 |
| Table 8 | 模型结构消融 |
| Table 9 | AAindex 维度敏感性 |
| Table 10 | CNN kernel / CNN layer 敏感性 |
| Table 11 | 序列长度与截断策略 |
| Table 12 | 推理时间和资源消耗 |

### 12.2 图片
| 图编号 | 图名 |
|---|---|
| Fig. 1 | MICA-ACP 模型结构 |
| Fig. 2 | 六个数据集序列长度分布 |
| Fig. 3 | Main/Alternate 上 ROC 曲线 |
| Fig. 4 | Main/Alternate 上 PR 曲线 |
| Fig. 5 | 六数据集雷达图 |
| Fig. 6 | AAindex 维度敏感性折线图 |
| Fig. 7 | 序列长度敏感性折线图 |
| Fig. 8 | t-SNE/UMAP 表征可视化 |
| Fig. 9 | attention/fusion gate 热图 |
| Fig. 10 | hard negative 权重分布 |


## 13. 当前代码建议补充的实验开关
为了更方便完成上述实验，建议给 `config.py` 和模型增加以下开关：
| 开关 | 作用 |
|---|---|
| `--Lmax` | 控制最大序列长度 |
| `--truncate_strategy` | 控制 head/tail/center/head_tail 截断 |
| `--disable_blosum` | 去掉 BLOSUM |
| `--disable_esm` | 去掉 ESM 分支 |
| `--disable_p0` | 去掉 P0 分支 |
| `--disable_token_cnn` | 去掉 token-CNN 分支 |
| `--disable_cross_attention` | 去掉 P0-ESM cross-attention |
| `--fusion_type` | 选择 gated_add/concat/add/swiglu |
| `--post_sequence_model` | 选择 none/bigru/lstm/gru/transformer/cnn/attention |
| `--lambda_hn` | 暴露 hard negative 强度 |
| `--topk_pos` | 暴露 H_embed top-k |
| `--rho` | 暴露 loss memory EMA |
| `--tau_loss` | 暴露 q_loss 裁剪阈值 |
| `--alpha` | 暴露 H_embed 与 q_loss 混合权重 |
| `--beta` | 暴露交互项权重 |
