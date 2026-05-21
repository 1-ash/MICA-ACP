# MICA-ACP 模型与实验总结
## 1. 模型总体思路
MICA-ACP 是一个用于抗癌肽（ACP）二分类任务的多源特征融合模型。模型同时使用三类输入信息：
1. 氨基酸 token 序列，用于学习局部序列模式。
2. 手工理化先验特征 `P0`，由 BLOSUM、AAindex 和 AAC20 组成。
3. ESM-C 预训练蛋白语言模型 embedding，用于提供语义级蛋白序列表征。

整体流程可以概括为：
```text
氨基酸序列
├── token_ids -> Embedding -> 多尺度 CNN 局部分支
├── P0(BLOSUM + AAindex + AAC20) -> 先验投影分支
└── ESM-C embedding -> 语义投影分支

P0 分支与 ESM 分支通过 Cross-Attention 交互
Cross-Attention 结果与 CNN 局部分支通过门控融合
融合后的序列表征经过 BiGRU
再经过注意力池化和 MLP 分类头输出 ACP 概率
```
该设计的直观目标是：用 ESM-C 捕获深层语义信息，用 BLOSUM/AAindex/AAC20 提供显式理化先验，用多尺度 CNN 捕获局部 motif，再通过注意力和门控机制完成融合。

## 2. 输入特征设置
本次实验最大序列长度为：
```text
Lmax = 50
```

序列长度超过 50 的样本会被截断，短序列会 padding 到 50。每个 batch 的主要输入张量如下：
| 输入 | 形状 | 说明 |
|---|---:|---|
| `token_ids` | `[B, 50]` | 氨基酸 token 编号，padding id 为 0 |
| `valid_mask` | `[B, 50]` | 有效残基位置 mask |
| `p0` | `[B, 50, 100]` | 手工先验特征 |
| `esm_h` | `[B, 50, 1152]` | ESM-C 600M embedding |

### 3.1 P0 特征组成
本次实验中 `P0` 总维度为：
```text
p0_dim = 100
```

由以下三部分组成：
| 特征来源 | 维度 | 说明 |
|---|---:|---|
| BLOSUM | 20 | 每个残基的 BLOSUM 表征 |
| AAindex | 60 | 通过 mRMR 筛选出的 60 个理化尺度 |
| AAC20 | 20 | 全序列氨基酸组成特征，并广播到每个残基位置 |
| 合计 | 100 | `20 + 60 + 20` |

### 3.2 ESM-C 特征
本次实验使用 ESM-C 600M 作为预训练蛋白语言模型特征来源：
```text
esm_model_name = ESM-C-600M
esm_dim = 1152
```

ESM-C embedding 会缓存到 `cache` 目录，训练时直接读取缓存后的 `[Lmax, 1152]` 表征。

## 4. AAindex mRMR 筛选
本次实验启用了 AAindex 的 mRMR 筛选：
```text
select_aaindex_mrmr = True
aaindex_dim = 60
```

筛选配置如下：

| 参数 | 值 |
|---|---:|
| `aaindex_selection_repeats` | 20 |
| `aaindex_selection_sample_fraction` | 0.8 |
| `aaindex_redundancy_threshold` | 0.95 |

筛选结果保存在：

```text
logs/main_k60_mRMR_lr5e-5_seed42_test2/aaindex_mrmr_selection.json
```

最终选出的 60 个 AAindex 索引为：

```text
74, 243, 550, 330, 299, 419, 93, 229, 237, 16,
144, 468, 185, 404, 326, 244, 497, 505, 194, 418,
493, 24, 284, 64, 495, 190, 75, 331, 385, 230,
375, 441, 273, 228, 19, 370, 490, 86, 269, 15,
279, 491, 172, 400, 412, 257, 218, 372, 123, 124,
378, 451, 376, 478, 25, 41, 395, 480, 449, 334
```

筛选出的 AAindex code 为：

```text
FASG760104, PONP800104, KARS160120, RICJ880110, RACS820101,
AURR980118, FINA910103, PALJ810108, PALJ810116, BUNA790102,
KHAG800101, FUKS010109, NAGK730101, AURR980103, RICJ880106,
PONP800105, GEOR030108, DIGM050101, NAKH900107, AURR980117,
GEOR030104, CHAM830102, QIAN880128, DAYM780201, GEOR030106,
NAKH900103, FASG760105, RICJ880111, WERD780103, PALJ810109,
VASM830101, KUMS000103, QIAN880117, PALJ810107, BURA740102,
TANS770106, GEOR030101, FAUJ880110, QIAN880113, BUNA790101,
QIAN880123, GEOR030102, MAXF760103, ZIMJ680104, AURR980111,
QIAN880101, OOBM850102, TANS770108, ISOY800106, ISOY800107,
VELV850101, NADH010107, VASM830102, WILM950102, CHAM830103,
CHOP780205, YUTK870104, WILM950104, NADH010105, RICJ880114
```

日志中将 60 个 AAindex 分为四类，每类 15 个：

| 分组 | 含义 | 数量 |
|---|---|---:|
| `charge` | 电荷相关理化性质 | 15 |
| `hydrophobic` | 疏水性相关性质 | 15 |
| `amphipathic` | 两亲性相关性质 | 15 |
| `structure` | 结构倾向相关性质 | 15 |

这说明本次实验不是直接使用全部 AAindex，而是通过 mRMR 保留了与标签相关、同时冗余度较低的一组理化尺度。

## 5. 模型结构细节

本次模型统一隐藏维度为：

```text
d = 256
```

### 5.1 ESM 语义分支

ESM-C 输出维度为 1152，模型首先将其投影到统一隐藏维度 256：

```text
ESM embedding [B, 50, 1152]
-> Linear(1152, 256)
-> LayerNorm(256)
```

对应代码模块：

```text
self.esm_proj = nn.Linear(cfg.esm_dim, cfg.d)
self.esm_norm = nn.LayerNorm(cfg.d)
```

### 5.2 P0 先验分支

P0 输入维度为 100。本次实验未启用 prior bottleneck：

```text
use_prior_bottleneck = False
```

因此 P0 分支结构为：

```text
P0 [B, 50, 100]
-> LayerNorm(100)
-> ScaleDropout(0.15)
-> Linear(100, 256)
-> Dropout(0.3)
-> LayerNorm(256)
```

其中 `ScaleDropout` 是按 P0 特征维度进行 dropout，用于降低模型对单个 AAindex scale 的过度依赖。

### 5.3 P0 与 ESM 的 Cross-Attention

P0 分支和 ESM 分支都被投影到 256 维后，进入 cross-attention：

```text
MultiheadAttention(
  embed_dim = 256,
  num_heads = 8,
  dropout = 0.3,
  batch_first = True
)
```

本次实验的注意力方向为：

```text
cross_attention_direction = prior_to_semantic
```

也就是：

```text
query = prior
key = semantic
value = semantic
residual = semantic
```

模型会用一个可学习门控参数控制 cross-attention 上下文注入残差分支的强度：

```text
gate_init = -2.2
sigmoid(-2.2) ≈ 0.10
```

这种初始化让 cross-attention 在训练初期以较小强度介入，避免一开始就强烈扰动 ESM 语义表征。

### 5.4 Token 多尺度 CNN 局部分支

token 分支首先将氨基酸 token 映射为 256 维 embedding：

```text
Embedding(21, 256, padding_idx=0)
-> LayerNorm(256)
```

随后使用 2 层 `MultiScaleCNNBlock`：

```text
num_cnn_layers = 2
cnn_kernels = [3, 5, 7]
```

每个 CNN block 包含 3 个不同卷积核大小的并行分支：

```text
LayerNorm
-> Conv1d(k=3)
-> Conv1d(k=5)
-> Conv1d(k=7)
-> concat
-> 1x1 Conv 投影回 256 维
-> GELU
-> Dropout(0.3)
-> residual add
```

多尺度卷积的作用是同时捕获不同长度的局部序列模式，例如短 motif 和稍长的局部残基组合。

### 5.5 语义分支与局部分支的门控融合

Cross-Attention 后的语义表征 `semantic_ca` 与 CNN 局部分支输出 `local` 通过 `GatedAddFusion` 融合：

```text
gate = sigmoid(Linear([semantic_ca, local]))
fused = gate * semantic_ca + (1 - gate) * local
```

本次融合层使用：

```text
branch_dropout = 0.1
dropout = 0.3
```

`branch_dropout` 会以样本为单位随机弱化 semantic 或 local 分支，有助于避免模型完全依赖单一特征来源。

### 5.6 融合后 BiGRU 序列建模

本次实验启用了融合后的 BiGRU：

```text
use_post_fusion_bigru = True
bigru_hidden = 64
bigru_layers = 1
bigru_dropout = 0.0
```

结构为：

```text
fused [B, 50, 256]
-> BiGRU(input_size=256, hidden_size=64, bidirectional=True)
-> Linear(128, 256)
-> LayerNorm(256)
```

由于是双向 GRU，输出维度为 `2 * 64 = 128`，再通过线性层投影回 256 维。

### 5.7 注意力池化与分类头

最后对序列维度做注意力加权池化：

```text
AttentionWeightedMeanPooling(256)
```

池化后得到 `[B, 256]` 的样本级表示，再进入分类头：

```text
Linear(256, 64)
-> LayerNorm(64)
-> GELU
-> Dropout(0.3)
-> Linear(64, 1)
```

模型输出一个 logit，经过 sigmoid 后得到样本为 ACP 的概率。

## 6. 参数规模

从 `best.pt` 读取到模型总参数量为：

```text
1,683,145
```

主要模块参数量如下：

| 模块 | 参数量 |
|---|---:|
| 多尺度 CNN blocks | 785,406 |
| ESM projection | 295,168 |
| Cross-Attention | 263,168 |
| Gated fusion | 131,840 |
| BiGRU | 123,648 |
| BiGRU projection | 33,024 |
| P0 branch | 26,568 |
| Classifier | 16,641 |
| Token embedding | 5,376 |

可以看到，参数主要集中在多尺度 CNN、ESM 投影层和 cross-attention 模块。

## 7. 训练配置

本次实验的关键训练超参数如下：

| 参数 | 值 |
|---|---:|
| `optimizer` | AdamW |
| `lr` | 5e-5 |
| `weight_decay` | 0.01 |
| `batch_size` | 64 |
| `epochs` | 200 |
| `clip_grad_norm` | 1.0 |
| `early_stopping_patience` | 20 |
| `seed` | 42 |
| `use_amp` | False |
| `use_scheduler` | False |
| `dropout` | 0.3 |
| `threshold` | 0.5 |

正则化相关配置：

| 参数 | 值 |
|---|---:|
| `use_ema` | True |
| `ema_decay` | 0.995 |
| `ema_start_epoch` | 5 |
| `use_rdrop` | True |
| `lambda_rdrop` | 0.1 |
| `scale_dropout` | 0.15 |
| `branch_dropout` | 0.1 |

训练损失主体为 `BCEWithLogitsLoss`。当启用 R-Drop 时，同一个 batch 会进行两次前向传播，并对两次 sigmoid 概率加入一致性约束：

```text
loss = weighted_BCE + lambda_rdrop * rdrop_loss
```

其中：

```text
lambda_rdrop = 0.1
```

## 8. Hard Negative Mining / NHL 设置

本次实验启用了 Hard Negative Mining 机制，代码中对应 `HardNegativeMiner` 和 NHL 相关逻辑。

核心配置如下：

| 参数 | 值 |
|---|---:|
| `start_nhl_epoch` | 30 |
| `nhl_warmup_epochs` | 30 |
| `topk_pos` | 10 |
| `rho` | 0.9 |
| `tau_loss` | 0.95 |
| `alpha` | 0.5 |
| `beta` | 0.5 |
| `lambda_hn` | 1 |
| `w_hn_max` | 1.2 |
| `h_embed_update` | every_n_epochs |
| `h_embed_update_interval` | 1 |
| `h_embed_ema_decay` | 0.95 |

负样本权重形式大致为：

```text
weight = 1 + lambda_hn * gate * hardness
```

并限制最大负样本权重为：

```text
w_hn_max = 1.2
```

因此 hard negative 机制整体比较保守，不会让少数困难负样本在训练中获得过高权重。

`H_embed` 的更新策略为：

```text
h_embed_update = every_n_epochs
h_embed_update_interval = 1
h_embed_ema_decay = 0.95
```

也就是从 NHL 开始后，每个 epoch 更新一次 embedding hardness，并使用 EMA 动量进行平滑。

## 9. 数据集设置

本次实验使用的数据集如下：

| 数据集 | 样本数 | 正样本 | 负样本 | 长度范围 |
|---|---:|---:|---:|---|
| `ACP2_main_train.tsv` | 1378 | 689 | 689 | 3-50 |
| `ACP2_main_test.tsv` | 344 | 172 | 172 | 2-50 |

配置中：

```text
use_test_as_valid = True
```

这意味着训练过程中直接使用 test set 作为 valid set，用于选择 best checkpoint 和搜索最佳阈值。因此本次日志中的 `valid_metrics` 和 `test_metrics` 完全一致。

该设置可以用于复现实验或对齐某些 ACP 文献中的评估协议，但如果目标是严格评估泛化性能，建议额外划分独立 validation set，避免 test 集参与模型选择。

## 10. 最优 checkpoint 与训练过程

本次最优 checkpoint 为：

```text
checkpoints/main_k60_mRMR_lr5e-5_seed42_test2/best.pt
```

从 checkpoint 读取到：

```text
best epoch = 74
checkpoint_uses_ema = True
```

也就是说，保存的最优模型使用 EMA 平滑后的权重。

训练日志显示，epoch 74、75、76、77 的 `valid_MCC` 和 `valid_ACC` 相同，均达到该次实验的最高水平。由于保存逻辑是：

```text
valid_MCC 提升，或者 valid_MCC 相同但 valid_ACC 提升时，更新 best checkpoint
```

因此最早达到该最优水平的 epoch 74 被保存为 `best.pt`。

训练最终运行到 epoch 94 后触发 early stopping。

## 11. 最终指标

最终指标来自：

```text
logs/main_k60_mRMR_lr5e-5_seed42_test2/test_metrics.json
```

### 11.1 默认阈值 0.5

默认阈值下：

```text
threshold = 0.5
```

测试指标为：

| 指标 | 值 |
|---|---:|
| ACC | 0.8081 |
| Precision | 0.8011 |
| Sensitivity | 0.8198 |
| Specificity | 0.7965 |
| F1 | 0.8103 |
| AUC | 0.8751 |
| MCC | 0.6164 |
| Loss | 0.5808 |
| TP | 141 |
| TN | 137 |
| FP | 35 |
| FN | 31 |

### 11.2 最佳验证阈值

根据验证集搜索得到的最佳阈值为：

```text
best_valid_threshold = 0.5774918497
```

该阈值下的测试指标为：

| 指标 | 值 |
|---|---:|
| ACC | 0.8140 |
| Precision | 0.8140 |
| Sensitivity | 0.8140 |
| Specificity | 0.8140 |
| F1 | 0.8140 |
| AUC | 0.8751 |
| MCC | 0.6279 |
| Loss | 0.5808 |
| TP | 140 |
| TN | 140 |
| FP | 32 |
| FN | 32 |

相比默认阈值，最佳阈值主要带来如下变化：

| 指标 | 默认阈值 0.5 | 最佳阈值 0.5775 |
|---|---:|---:|
| ACC | 0.8081 | 0.8140 |
| MCC | 0.6164 | 0.6279 |
| Specificity | 0.7965 | 0.8140 |
| Sensitivity | 0.8198 | 0.8140 |

可以看到，最佳阈值略微降低了 Sensitivity，但提高了 Specificity、ACC 和 MCC，使正负类识别更加均衡。

## 12. 总结

本次 `main_k60_mRMR_lr5e-5_seed42_test2` 实验使用了一个多源信息融合的 ACP 分类模型。模型融合了：

1. ESM-C 600M 语义 embedding。
2. BLOSUM、mRMR-AAindex 和 AAC20 组成的显式理化先验。
3. token embedding 上的多尺度 CNN 局部模式。
4. P0 与 ESM 之间的 prior-to-semantic cross-attention。
5. 语义分支与局部分支之间的门控融合。
6. 融合后的 BiGRU 序列上下文建模。
7. 注意力池化与 MLP 分类头。

本次实验的最优模型出现在 epoch 74，使用 EMA 权重，模型参数量约为 168 万。在 ACP2 main test set 上，默认阈值下 MCC 为 0.6164，AUC 为 0.8751；使用验证集搜索得到的最佳阈值 0.5775 后，MCC 提升到 0.6279，ACC 提升到 0.8140。

需要在报告中明确说明的是：该实验设置了 `use_test_as_valid=True`，因此测试集同时参与了 best checkpoint 选择和最佳阈值搜索。若用于严格泛化性能评估，应额外引入独立验证集，或采用训练集内部验证划分来选择 checkpoint 和阈值。
