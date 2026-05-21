from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

# HuggingFace tokenizers can warn after DataLoader forks worker processes.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_TRAIN_TSV_NAME = "ACP2_main_train.tsv"  # 默认训练集文件名；未传 --train_tsv 时在 data_root 下查找。
DEFAULT_TEST_TSV_NAME = "ACP2_main_test.tsv"  # 默认测试集文件名；未传 --test_tsv 时在 data_root 下查找。

ESM_MODEL_REGISTRY_NAMES = {"esmc_300m", "esmc_600m"}  # 允许直接传入的 ESM-C 模型注册名。


AAINDEX_TOTAL_ROWS: int = 553  # AAINDEX.csv 中候选 scale 的总行数。

# 全 553 维候选 (无任何筛选,按 AAINDEX.csv 原始行序)
AAINDEX_FULL_INDICES: list[int] = list(range(AAINDEX_TOTAL_ROWS))  # 不做筛选时使用全部 AAindex 行。

# 按重要度排序的 60 列子集 (做小规模消融时使用)
AAINDEX_TOP60_INDICES: list[int] = [  # 预置的 AAindex 优先子集；用于默认 60 维或小规模消融。
    109, 268, 396,  92, 282,  14, 269, 220,  58, 307,
    332,  56, 327, 405, 415,  84, 472, 238, 485,  86,
    299, 187, 433, 292, 443, 266,  64, 103, 280, 340,
     48, 106, 257, 481,  91, 400, 267, 449, 217, 399,
    499,  54, 445,  23, 391, 314, 218,  40, 101, 291,
    473, 422, 226, 265, 442,  36, 414, 154, 426, 172,
]

AAINDEX_PRIORITIZED_INDICES: list[int] = AAINDEX_TOP60_INDICES + [  # 先放 Top60，再补齐剩余行，便于按 aaindex_dim 截断。
    idx for idx in AAINDEX_FULL_INDICES if idx not in set(AAINDEX_TOP60_INDICES)
]

# >>>>> 选择本轮使用的索引池 + 维度 (做消融实验主要修改这两行) <<<<<
# 用法 1 (当前): 重要度排序子集 (取前 N 个,N ∈ {32, 48, 60, 72})
#     DEFAULT_AAINDEX_INDICES = AAINDEX_PRIORITIZED_INDICES
#     DEFAULT_AAINDEX_DIM     = 60
# 用法 2: 全部 553 维
#     DEFAULT_AAINDEX_INDICES = AAINDEX_FULL_INDICES
#     DEFAULT_AAINDEX_DIM     = AAINDEX_TOTAL_ROWS    # = 553
# 用法 3: 关闭 AAINDEX 通道 (只用 BLOSUM62)
#     DEFAULT_AAINDEX_INDICES = AAINDEX_FULL_INDICES   # 任意池都行
#     DEFAULT_AAINDEX_DIM     = 0
DEFAULT_AAINDEX_INDICES: list[int] = AAINDEX_PRIORITIZED_INDICES  # 默认 AAindex 行索引池，sync_derived_dims 会按维度截断。
DEFAULT_AAINDEX_DIM: int = 60  # 默认实际使用的 AAindex scale 数量；0 表示关闭 AAindex 通道。
DEFAULT_AAC_DIM: int = 20  # AAC20 氨基酸组成先验维度；启用时广播拼接到每个残基的 P0。

# --- CNN 多尺度卷积核 --------------------------------------------------------
# 必须为奇数列表,长度 = 卷积分支数。
# 推荐做消融的取值: [3], [3,5], [3,5,7], [5,7,9] 等。
DEFAULT_CNN_KERNELS: list[int] = [3, 5, 7]  # 多尺度局部 CNN 的卷积核大小；必须为奇数以保持 same padding。

# CNN 堆叠层数 (每层包含 1 个 Conv + GeLU + Norm)
DEFAULT_NUM_CNN_LAYERS: int = 2  # MultiScaleCNNBlock 堆叠层数，用于 token embedding 的局部模式建模。

# --- Transformer / 注意力 ----------------------------------------------------
DEFAULT_NUM_HEADS: int = 8  # prior/semantic cross-attention 的注意力头数。
DEFAULT_HIDDEN_DIM: int = 256  # 模型统一隐藏维度 d；ESM、P0、token-CNN 都会投影到该维度。
DEFAULT_FFN_DIM: int = 512  # 历史保留字段；当前主线不使用，仅 dataclass d_ff 字段引用以兼容旧 ckpt/配置。

# --- 正则化 -----------------------------------------------------------------
DEFAULT_DROPOUT: float = 0.3  # 主干通用 dropout；用于 cross-attention、CNN block、融合层和分类头。
DEFAULT_USE_PRIOR_BOTTLENECK: bool = False  # P0 分支是否启用瓶颈层：False=单层 p0->d 投影；True=两层 p0->bottleneck->d。
DEFAULT_PRIOR_BOTTLENECK_DIM: int = 64  # use_prior_bottleneck=True 时的中间瓶颈维度，控制 P0 投影容量。
DEFAULT_PRIOR_DROPOUT: float = 0.3  # P0 先验分支投影后的 dropout（瓶颈层之后 / 单层投影之后均生效）。
DEFAULT_SCALE_DROPOUT: float = 0.15  # 对 P0 特征维度做整列 dropout，降低模型对单个 AAindex scale 的依赖。
DEFAULT_BRANCH_DROPOUT: float = 0.10  # GatedAddFusion 中按样本丢弃 semantic/local 分支的概率。
DEFAULT_CROSS_ATTENTION_DIRECTION: str = "prior_to_semantic"  # prior_to_semantic: P0 查询 ESM；semantic_to_prior: ESM 查询 P0。
DEFAULT_FUSION_TYPE: str = "gated_add"  # semantic/local 融合方式：gated_add / concat / add / swiglu。
DEFAULT_POST_SEQUENCE_MODEL: str = "bigru"  # 融合后的序列建模模块：none / bigru / lstm / gru / transformer / cnn / attention。
DEFAULT_USE_POST_FUSION_BIGRU: bool = True  # 融合后是否使用双向 GRU 做序列级上下文交互。
DEFAULT_BIGRU_HIDDEN: int = 64  # post-fusion BiGRU 单方向隐藏维度，输出会再投影回 d。
DEFAULT_BIGRU_LAYERS: int = 1  # post-fusion BiGRU 层数。
DEFAULT_BIGRU_DROPOUT: float = 0.0  # post-fusion BiGRU 层间 dropout；仅 bigru_layers > 1 时由 PyTorch 生效。


# =============================================================================
# 3. 输入特征维度 (一般不修改)
# =============================================================================
DEFAULT_LMAX: int = 50  # 肽序列最大保留长度；清洗、padding、P0 和 ESM cache 都使用该长度。
DEFAULT_TRUNCATE_STRATEGY: str = "head"  # 超长序列截断策略：head / tail / center / head_tail。
DEFAULT_BLOSUM_DIM: int = 20  # BLOSUM62 每个残基的固定 20 维先验。
DEFAULT_ESM_DIM: int = 1152  # ESM-C embedding 维度；必须与 esm_model_name/esm_model_dir 的实际输出一致。
DEFAULT_ESM_MODEL_NAME: str = "ESM-C-600M"  # ESM cache 元数据和模型维度校验使用的模型名。


# =============================================================================
# 4. 训练超参
# =============================================================================
DEFAULT_EPOCHS: int = 200  # 最大训练轮数；可被 early stopping 提前终止。
DEFAULT_BATCH_SIZE: int = 64  # train/valid/test DataLoader 的 batch size。
DEFAULT_LR: float = 5e-5  # AdamW 学习率。
DEFAULT_WEIGHT_DECAY: float = 1e-2  # AdamW 权重衰减；bias、norm 和 embedding 参数会被分到 no_decay 组。
DEFAULT_CLIP_GRAD_NORM: float = 1.0  # 梯度范数裁剪阈值；<=0 时不裁剪。
DEFAULT_EARLY_STOPPING_PATIENCE: int = 20  # 验证集 MCC/ACC 连续无提升多少轮后提前停止。
DEFAULT_SEED: int = 42  # 随机种子；控制数据切分、mRMR 重采样和训练随机性。
DEFAULT_VALID_RATIO: float = 0.2  # 仅当显式关闭 use_test_as_valid 时，从训练集分出的验证集比例。
DEFAULT_USE_TEST_AS_VALID: bool = True  # 默认直接把 test 集当验证集，用于选 best ckpt / 阈值；
DEFAULT_THRESHOLD: float = 0.5  # 默认分类阈值；用于训练日志、默认指标和推理输出。
DEFAULT_NUM_WORKERS: int = 2  # DataLoader 子进程数。
DEFAULT_USE_AMP: bool = False  # 是否在 CUDA 上启用自动混合精度训练。
DEFAULT_OPTIMIZER: str = "AdamW"  # 配置记录字段；当前 train.py 固定构造 AdamW。
DEFAULT_USE_SCHEDULER: bool = False  # 是否启用 ReduceLROnPlateau 学习率调度器。
DEFAULT_LR_SCHEDULER_MONITOR: str = "valid_loss"  # 调度器监控的验证指标，valid_ 前缀会被自动去掉。
DEFAULT_LR_SCHEDULER_MODE: str = "min"  # 调度器优化方向；loss 用 min，其它指标通常用 max。
DEFAULT_USE_EMA: bool = True  # 是否维护模型参数指数滑动平均，并用 EMA 权重评估/保存 best。
DEFAULT_EMA_DECAY: float = 0.995  # EMA 衰减系数，越大越平滑。
DEFAULT_EMA_START_EPOCH: int = 5  # 从第几轮开始更新 EMA。
DEFAULT_USE_RDROP: bool = True  # 是否启用双前向一致性正则。
DEFAULT_LAMBDA_RDROP: float = 0.1  # R-Drop 一致性损失权重。
DEFAULT_THRESHOLD_OBJECTIVE: str = "MCC"  # 阈值搜索目标；当前只支持 MCC。
DEFAULT_THRESHOLD_MIN_SPECIFICITY: float = 0  # 验证集阈值搜索时要求的最低 Specificity。
DEFAULT_THRESHOLD_SEARCH_MIN: float = 0.30  # 最优阈值搜索下界。
DEFAULT_THRESHOLD_SEARCH_MAX: float = 0.70  # 最优阈值搜索上界。
DEFAULT_THRESHOLD_SEARCH_STEP: float = 0.005  # 阈值网格搜索步长，同时会加入预测概率中点候选。


# =============================================================================
# 5. Hard Negative Mining 参数
# =============================================================================
DEFAULT_TOPK_POS: int = 10  # H_embed 中每个负样本取最相似的 top-k 正样本求平均相似度。
DEFAULT_RHO: float = 0.9  # hard negative loss memory 的 EMA 衰减系数。
DEFAULT_TAU_LOSS: float = 0.95  # loss 排名分位数截断上限，用于稳定 q_loss。
DEFAULT_ALPHA: float = 0.5  # combined_hardness 中 H_embed 与 q_loss 的线性混合权重。
DEFAULT_BETA: float = 0.5  # combined_hardness 中 H_embed*q_loss 交互项权重。
DEFAULT_LAMBDA_HN: float = 1 # hard negative 样本权重增益，权重为 1 + lambda_hn * gate * hardness。
DEFAULT_GATE_INIT: float = -2.2  # cross-attention 残差门控初值，sigmoid 后约 0.1。
DEFAULT_START_NHL_EPOCH: int = 30  # 从第几轮开始计算 H_embed 并启用 NHL hard-negative 加权。
DEFAULT_NHL_WARMUP_EPOCHS: int = 30  # NHL gate 从 0 线性升到 1 的 warmup 轮数。
DEFAULT_E_HN_WARM: int = 10  # 历史保留字段；当前训练循环已不再使用，保留以兼容旧配置/旧 checkpoint。
DEFAULT_H_EMBED_UPDATE: str = "every_n_epochs"  # H_embed 更新策略：start_nhl_epoch_once / every_epoch / every_n_epochs。
# 默认 every_n_epochs + interval=1 + ema_decay=0.9：
# 每个 epoch 用当前表征算一次 new_h_embed，再做 h_embed = decay*old + (1-decay)*new 的动量混合，
# 既能持续跟随表征漂移、又不会被单轮表征噪声拉动；冷启动（首次进入 NHL）时强制 decay=0 直接覆盖。
DEFAULT_H_EMBED_UPDATE_INTERVAL: int = 1  # h_embed_update=every_n_epochs 时的更新间隔，单位 epoch。
DEFAULT_H_EMBED_EMA_DECAY: float = 0.95  # H_embed EMA 动量；0 表示直接覆盖（旧行为），(0,1) 表示动量混合。
DEFAULT_W_HN_MAX: float = 1.2  # hard negative 样本权重上限，防止少数负样本主导训练。
DEFAULT_AAINDEX_SELECTION_REPEATS: int = 20  # mRMR 稳定筛选的重采样次数。
DEFAULT_AAINDEX_SELECTION_SAMPLE_FRACTION: float = 0.8  # 每次 mRMR 重采样使用的训练样本比例。
DEFAULT_AAINDEX_REDUNDANCY_THRESHOLD: float = 0.95  # AAindex scale 相关性超过该阈值时视为冗余。


# =============================================================================
# 6. SwanLab 实验追踪
# =============================================================================
DEFAULT_USE_SWANLAB: bool = True  # 是否启用 SwanLab 实验追踪。
DEFAULT_SWANLAB_PROJECT: str = "MICA-ACP-V1"  # SwanLab 默认项目名。
DEFAULT_SWANLAB_API_KEY: str | None = "pWSSR0qR01AUw2IeRUteN"  # SwanLab 默认 API key，可被环境变量/命令行覆盖。


# =============================================================================
# 7. 高级:环境变量映射 (一般无需关心)
# =============================================================================
PATH_ENV_VARS = {
    "project_root": "GLACE_PROJECT_ROOT",
    "train_tsv_path": "GLACE_TRAIN_TSV",
    "valid_tsv_path": "GLACE_VALID_TSV",
    "test_tsv_path": "GLACE_TEST_TSV",
    "aaindex_csv_path": "GLACE_AAINDEX_CSV",
    "blosum_csv_path": "GLACE_BLOSUM_CSV",
    "esm_model_dir": "GLACE_ESM_MODEL_DIR",
    "cache_dir": "GLACE_CACHE_DIR",
    "checkpoint_dir": "GLACE_CHECKPOINT_DIR",
    "log_dir": "GLACE_LOG_DIR",
}
PATH_FIELDS = tuple(key for key in PATH_ENV_VARS if key != "project_root")


# =============================================================================
# 8. 路径解析与默认工厂  (实现细节,可不阅读)
# =============================================================================
def _normalise_esm_registry_name(value: str | Path) -> str | None:
    text = str(value).strip().lower().replace("-", "_")
    return text if text in ESM_MODEL_REGISTRY_NAMES else None


def _path_from(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base / path


def _configured_project_root() -> Path:
    value = os.environ.get("GLACE_PROJECT_ROOT")
    if value:
        return _path_from(value, PROJECT_ROOT)
    return PROJECT_ROOT


def _path_string(value: str | Path, base: Path | None = None) -> str:
    return str(_path_from(value, base or _configured_project_root()))


def _project_root_string(value: str | Path) -> str:
    return str(_path_from(value, PROJECT_ROOT))


def _default_data_root() -> Path:
    value = os.environ.get("GLACE_DATA_ROOT")
    if value:
        return _path_from(value, _configured_project_root())
    return _configured_project_root() / "data" / "ACP_dataset"


def _default_train_tsv_path() -> str:
    value = os.environ.get("GLACE_TRAIN_TSV")
    if value:
        return _path_string(value)
    return str(_default_data_root() / DEFAULT_TRAIN_TSV_NAME)


def _default_valid_tsv_path() -> str | None:
    value = os.environ.get("GLACE_VALID_TSV")
    if value:
        return _path_string(value)
    return None


def _default_test_tsv_path() -> str:
    value = os.environ.get("GLACE_TEST_TSV")
    if value:
        return _path_string(value)
    return str(_default_data_root() / DEFAULT_TEST_TSV_NAME)


def _default_aaindex_path() -> str:
    value = os.environ.get("GLACE_AAINDEX_CSV")
    if value:
        return _path_string(value)
    return str(_configured_project_root() / "AAINDEX.csv")


def _default_blosum_path() -> str:
    value = os.environ.get("GLACE_BLOSUM_CSV")
    if value:
        return _path_string(value)
    root = _configured_project_root()
    for name in ("BLOSUM62.csv", "BLOSUM6.csv"):
        path = root / name
        if path.exists():
            return str(path)
    return str(root / "BLOSUM62.csv")


def _default_esm_model_dir() -> str:
    value = os.environ.get("GLACE_ESM_MODEL_DIR")
    if value:
        registry_name = _normalise_esm_registry_name(value)
        if registry_name is not None:
            return registry_name
        return _path_string(value)
    root = _configured_project_root()
    for candidate in (root / "esmc_600m", root.parent / "esmc_600m"):
        if candidate.exists():
            return str(candidate)
    return str(root / "esmc_600m")


def _try_relative_parts(path_value: str, old_root: str) -> tuple[str, ...] | None:
    try:
        return Path(path_value).relative_to(Path(old_root)).parts
    except ValueError:
        pass
    for path_cls in (PureWindowsPath, PurePosixPath):
        try:
            return path_cls(path_value).relative_to(path_cls(old_root)).parts
        except ValueError:
            continue
    return None


def _rebase_path(path_value: str, old_root: str, new_root: Path) -> str | None:
    parts = _try_relative_parts(path_value, old_root)
    if parts is None:
        return None
    return str(new_root.joinpath(*parts))


def _parse_int_list(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, str):
        if not value.strip():
            return []
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(item) for item in value]


def _parse_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


# =============================================================================
# 9. 配置数据类  (训练 / 推理统一入口)
# =============================================================================
@dataclass
class MICAConfig:
    # ---- 路径 ----
    project_root: str = field(default_factory=lambda: str(_configured_project_root()))  # 项目根目录；相对路径都按它解析。
    train_tsv_path: str = field(default_factory=_default_train_tsv_path)  # 训练集 TSV，需包含 text/label 列。
    valid_tsv_path: str | None = field(default_factory=_default_valid_tsv_path)  # 可选验证集 TSV；为空则按 valid_ratio 切分。
    test_tsv_path: str = field(default_factory=_default_test_tsv_path)  # 测试集 TSV，用于训练结束后的最终评估。
    aaindex_csv_path: str = field(default_factory=_default_aaindex_path)  # AAindex scale 表路径。
    blosum_csv_path: str = field(default_factory=_default_blosum_path)  # BLOSUM62 表路径。
    esm_model_dir: str = field(default_factory=_default_esm_model_dir)  # ESM-C 本地目录或注册名。
    cache_dir: str = field(default_factory=lambda: str(_configured_project_root() / "cache"))  # ESM embedding 缓存目录。
    checkpoint_dir: str = field(default_factory=lambda: str(_configured_project_root() / "checkpoints"))  # best/last checkpoint 输出目录。
    log_dir: str = field(default_factory=lambda: str(_configured_project_root() / "logs"))  # train_log 与 test_metrics 输出目录。

    # ---- 输入特征维度 ----
    Lmax: int = DEFAULT_LMAX  # 最大序列长度；超长序列清洗时截断，短序列 padding 到该长度。
    truncate_strategy: str = DEFAULT_TRUNCATE_STRATEGY  # 超长序列截断策略。
    blosum_dim: int = DEFAULT_BLOSUM_DIM  # BLOSUM 残基先验维度，通常固定为 20。
    esm_dim: int = DEFAULT_ESM_DIM  # ESM embedding 最后一维，需与所选 ESM-C 模型一致。
    esm_model_name: str = DEFAULT_ESM_MODEL_NAME  # ESM cache 元数据和模型规格校验使用。

    # ---- 消融:AAINDEX ----
    aaindex_dim: int = DEFAULT_AAINDEX_DIM  # 实际使用的 AAindex scale 数；会截断 aaindex_indices。
    aaindex_indices: list[int] = field(default_factory=lambda: list(DEFAULT_AAINDEX_INDICES))  # AAindex CSV 行索引池。
    use_aac20_prior: bool = True  # 是否把 AAC20 全序列氨基酸组成拼入每个残基的 P0。
    aac_dim: int = DEFAULT_AAC_DIM  # AAC20 维度，固定为 20；关闭 AAC20 时不计入 p0_dim。
    select_aaindex_mrmr: bool = False  # 训练开始前是否用训练集标签重新做 AAindex mRMR 筛选。
    aaindex_selection_repeats: int = DEFAULT_AAINDEX_SELECTION_REPEATS  # mRMR 重采样次数。
    aaindex_selection_sample_fraction: float = DEFAULT_AAINDEX_SELECTION_SAMPLE_FRACTION  # 每次 mRMR 抽样比例。
    aaindex_redundancy_threshold: float = DEFAULT_AAINDEX_REDUNDANCY_THRESHOLD  # mRMR 前的 scale 冗余过滤阈值。
    disable_blosum: bool = False  # 消融开关：关闭 BLOSUM 先验。
    disable_esm: bool = False  # 消融开关：关闭 ESM-C 语义分支。
    disable_p0: bool = False  # 消融开关：关闭 P0 理化先验分支。
    disable_token_cnn: bool = False  # 消融开关：关闭 token embedding/CNN 局部分支。
    p0_dim: int = DEFAULT_BLOSUM_DIM + DEFAULT_AAINDEX_DIM + DEFAULT_AAC_DIM  # 派生维度；sync_derived_dims 自动重算。

    # ---- 消融:架构 ----
    d: int = DEFAULT_HIDDEN_DIM  # 模型主隐藏维度；ESM、P0、CNN 和分类头共享该表示宽度。
    num_heads: int = DEFAULT_NUM_HEADS  # cross-attention 头数，需能整除 d。
    cnn_kernels: list[int] = field(default_factory=lambda: list(DEFAULT_CNN_KERNELS))  # 多尺度 CNN 核大小列表。
    num_cnn_layers: int = DEFAULT_NUM_CNN_LAYERS  # 多尺度 CNN block 堆叠数量。
    d_ff: int = DEFAULT_FFN_DIM  # 预留字段；当前模型未使用，保留给历史配置/后续 FFN 扩展。
    dropout: float = DEFAULT_DROPOUT  # 主干 dropout 概率。
    gate_init: float = DEFAULT_GATE_INIT  # cross-attention 残差门控 logit 初值。
    use_prior_bottleneck: bool = DEFAULT_USE_PRIOR_BOTTLENECK  # 是否启用 P0 分支的中间瓶颈层；False 时为单层 p0->d 投影。
    prior_bottleneck_dim: int = DEFAULT_PRIOR_BOTTLENECK_DIM  # use_prior_bottleneck=True 时的瓶颈维度，需 >0 且 <= d。
    prior_dropout: float = DEFAULT_PRIOR_DROPOUT  # P0 投影后、LayerNorm 前的 dropout。
    scale_dropout: float = DEFAULT_SCALE_DROPOUT  # P0 特征维度 dropout 概率。
    cross_attention_direction: str = DEFAULT_CROSS_ATTENTION_DIRECTION  # 控制 P0/ESM 哪个做 query。
    disable_cross_attention: bool = False  # 消融开关：关闭 P0-ESM cross-attention。
    branch_dropout: float = DEFAULT_BRANCH_DROPOUT  # semantic/local 融合前按分支 dropout 的概率。
    fusion_type: str = DEFAULT_FUSION_TYPE  # semantic/local 融合方式。
    post_sequence_model: str = DEFAULT_POST_SEQUENCE_MODEL  # 融合后的序列建模模块。
    use_post_fusion_bigru: bool = DEFAULT_USE_POST_FUSION_BIGRU  # 融合后是否启用 BiGRU。
    bigru_hidden: int = DEFAULT_BIGRU_HIDDEN  # BiGRU 单方向隐藏维度。
    bigru_layers: int = DEFAULT_BIGRU_LAYERS  # BiGRU 层数。
    bigru_dropout: float = DEFAULT_BIGRU_DROPOUT  # BiGRU 层间 dropout，仅多层时生效。

    # ---- Hard Negative Mining ----
    topk_pos: int = DEFAULT_TOPK_POS  # H_embed 用每个负样本最相近的 top-k 正样本相似度。
    rho: float = DEFAULT_RHO  # 负样本 BCE loss 记忆的 EMA 衰减。
    tau_loss: float = DEFAULT_TAU_LOSS  # q_loss 分位数裁剪阈值。
    alpha: float = DEFAULT_ALPHA  # H_embed 与 q_loss 的混合权重。
    beta: float = DEFAULT_BETA  # H_embed 和 q_loss 交互项权重。
    lambda_hn: float = DEFAULT_LAMBDA_HN  # hard negative 加权强度。
    E_HN_warm: int = DEFAULT_E_HN_WARM  # 旧字段；当前训练循环不读取，保留兼容旧配置。
    start_nhl_epoch: int = DEFAULT_START_NHL_EPOCH  # NHL hard-negative 机制开始生效的 epoch。
    nhl_warmup_epochs: int = DEFAULT_NHL_WARMUP_EPOCHS  # NHL 样本权重 gate 的线性 warmup 长度。
    h_embed_update: str = DEFAULT_H_EMBED_UPDATE  # H_embed 更新频率策略。
    h_embed_update_interval: int = DEFAULT_H_EMBED_UPDATE_INTERVAL  # every_n_epochs 策略的更新间隔。
    h_embed_ema_decay: float = DEFAULT_H_EMBED_EMA_DECAY  # H_embed 动量混合系数；0 退化为直接覆盖。
    w_hn_max: float = DEFAULT_W_HN_MAX  # hard negative 样本权重上限。

    # ---- 训练 ----
    optimizer: str = DEFAULT_OPTIMIZER  # 记录/兼容字段；当前 train.py 固定使用 AdamW。
    lr: float = DEFAULT_LR  # AdamW 初始学习率。
    weight_decay: float = DEFAULT_WEIGHT_DECAY  # AdamW 权重衰减。
    batch_size: int = DEFAULT_BATCH_SIZE  # DataLoader batch size。
    epochs: int = DEFAULT_EPOCHS  # 最大训练 epoch。
    clip_grad_norm: float = DEFAULT_CLIP_GRAD_NORM  # 梯度裁剪阈值。
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE  # 验证指标无提升的容忍 epoch 数。
    seed: int = DEFAULT_SEED  # 全局随机种子。
    valid_ratio: float = DEFAULT_VALID_RATIO  # 仅当显式关闭 use_test_as_valid 时才会从训练集中切分该比例做验证。
    positive_label_value: int = 1  # TSV 中哪个原始 label 值视为正类。
    threshold: float = DEFAULT_THRESHOLD  # 默认二分类阈值。
    num_workers: int = DEFAULT_NUM_WORKERS  # DataLoader worker 数。
    pin_memory: bool = True  # CUDA 训练时 DataLoader 是否启用 pin_memory。
    use_amp: bool = DEFAULT_USE_AMP  # CUDA 上是否使用 AMP。
    resume_path: str | None = None  # checkpoint 路径；非空时恢复模型、优化器、EMA、miner 状态。
    force_rebuild_cache: bool = False  # 是否忽略已有 ESM cache 并重新生成。
    use_test_as_valid: bool = DEFAULT_USE_TEST_AS_VALID  # 默认 True：直接用 test 集选 best ckpt / 阈值（对齐 ACP SOTA 做法）。
    use_scheduler: bool = DEFAULT_USE_SCHEDULER  # 是否启用 ReduceLROnPlateau。
    lr_scheduler_monitor: str = DEFAULT_LR_SCHEDULER_MONITOR  # 调度器监控指标。
    lr_scheduler_mode: str = DEFAULT_LR_SCHEDULER_MODE  # 调度器优化方向。
    use_ema: bool = DEFAULT_USE_EMA  # 是否使用参数 EMA。
    ema_decay: float = DEFAULT_EMA_DECAY  # EMA 衰减系数。
    ema_start_epoch: int = DEFAULT_EMA_START_EPOCH  # 开始更新 EMA 的 epoch。
    use_rdrop: bool = DEFAULT_USE_RDROP  # 是否开启双前向 R-Drop。
    lambda_rdrop: float = DEFAULT_LAMBDA_RDROP  # R-Drop 损失权重。
    threshold_objective: str = DEFAULT_THRESHOLD_OBJECTIVE  # 阈值搜索目标，目前只允许 MCC。
    threshold_min_specificity: float = DEFAULT_THRESHOLD_MIN_SPECIFICITY  # 搜索阈值时的 Specificity 下限。
    threshold_search_min: float = DEFAULT_THRESHOLD_SEARCH_MIN  # 阈值搜索范围下界。
    threshold_search_max: float = DEFAULT_THRESHOLD_SEARCH_MAX  # 阈值搜索范围上界。
    threshold_search_step: float = DEFAULT_THRESHOLD_SEARCH_STEP  # 阈值网格步长。

    # ---- 实验隔离 ----
    # 当 run_name 设置后,checkpoint_dir/log_dir 会被自动包到同名子目录里,
    # 不同消融实验互不覆盖。留空则在 train.py 启动时根据关键超参 + 时间戳自动生成。
    run_name: str | None = None  # 实验名；为空时按关键超参和时间戳自动生成。

    # ---- SwanLab ----
    use_swanlab: bool = DEFAULT_USE_SWANLAB  # 是否启用 SwanLab 日志追踪。
    swanlab_project: str = DEFAULT_SWANLAB_PROJECT  # SwanLab 项目名。
    swanlab_experiment: str | None = None  # SwanLab run 名；为空时跟随 run_name。
    swanlab_workspace: str | None = None  # SwanLab workspace/组织名。
    swanlab_mode: str | None = None  # SwanLab 运行模式：cloud/local/offline/disabled。
    swanlab_api_key: str | None = DEFAULT_SWANLAB_API_KEY  # SwanLab API key，可被环境变量覆盖。
    swanlab_description: str | None = None  # SwanLab run 描述。
    swanlab_tags: list[str] = field(default_factory=list)  # SwanLab 标签列表。
    swanlab_logdir: str | None = None  # SwanLab 本地日志目录。

    # ---------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.sync_derived_dims()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "MICAConfig":
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in values.items() if k in valid_keys}
        cfg = cls(**filtered)
        if "aaindex_indices" in values:
            cfg.aaindex_indices = _parse_int_list(values["aaindex_indices"])
            if "aaindex_dim" not in values:
                cfg.aaindex_dim = len(cfg.aaindex_indices)
        cfg.rebase_missing_project_paths(values.get("project_root"))
        cfg.apply_environment_overrides()
        cfg.sync_derived_dims()
        return cfg

    def ensure_dirs(self) -> None:
        for path in (self.cache_dir, self.checkpoint_dir, self.log_dir):
            Path(path).mkdir(parents=True, exist_ok=True)

    def auto_run_name(self) -> str:
        """根据消融关键超参 + 时间戳生成一个唯一 run_name."""
        from datetime import datetime

        kernels = "-".join(str(k) for k in self.cnn_kernels) or "none"
        lr_str = f"{self.lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        parts = [
            f"aaidx{self.aaindex_dim}",
            f"p0{self.p0_dim}",
            f"k{kernels}",
            f"cnnL{self.num_cnn_layers}",
            f"len{self.Lmax}",
            f"tr{self.truncate_strategy}",
            f"fus{self.fusion_type}",
            f"post{self.post_sequence_model}",
            f"h{self.num_heads}",
            f"d{self.d}",
            f"bs{self.batch_size}",
            f"lr{lr_str}",
            f"seed{self.seed}",
            datetime.now().strftime("%Y%m%d-%H%M%S"),
        ]
        return "_".join(parts)

    def resolve_run_dirs(self) -> None:
        """如果 run_name 已设置,把 checkpoint_dir / log_dir 各自包到同名子目录里 (幂等)."""
        if not self.run_name:
            return
        ckpt_path = Path(self.checkpoint_dir)
        if ckpt_path.name != self.run_name:
            self.checkpoint_dir = str(ckpt_path / self.run_name)
        log_path = Path(self.log_dir)
        if log_path.name != self.run_name:
            self.log_dir = str(log_path / self.run_name)

    def sync_derived_dims(self) -> None:
        """根据 aaindex_dim 截取 aaindex_indices,并同步 p0_dim 等派生维度."""
        if self.Lmax <= 0:
            raise ValueError(f"Lmax must be positive, got {self.Lmax}")
        if self.truncate_strategy not in {"head", "tail", "center", "head_tail"}:
            raise ValueError(
                "truncate_strategy must be one of head, tail, center, head_tail, "
                f"got {self.truncate_strategy!r}"
            )
        self.aaindex_indices = _parse_int_list(self.aaindex_indices)
        if self.aaindex_dim < 0:
            raise ValueError(f"aaindex_dim must be non-negative, got {self.aaindex_dim}")
        if self.aaindex_dim > len(self.aaindex_indices):
            raise ValueError(
                "aaindex_dim cannot exceed the number of configured aaindex_indices: "
                f"aaindex_dim={self.aaindex_dim}, len(indices)={len(self.aaindex_indices)}"
            )
        self.aaindex_indices = self.aaindex_indices[: self.aaindex_dim]
        if self.aac_dim != 20:
            raise ValueError(f"aac_dim must stay 20 for AAC20, got {self.aac_dim}")
        self.blosum_dim = 0 if self.disable_blosum else DEFAULT_BLOSUM_DIM
        aac_prior_dim = self.aac_dim if self.use_aac20_prior else 0
        self.p0_dim = 0 if self.disable_p0 else self.blosum_dim + self.aaindex_dim + aac_prior_dim
        if self.use_aac20_prior and not self.disable_p0 and self.p0_dim != self.blosum_dim + self.aaindex_dim + self.aac_dim:
            raise ValueError(
                "p0_dim must equal blosum_dim + aaindex_dim + aac_dim when AAC20 prior is enabled: "
                f"p0_dim={self.p0_dim}, aaindex_dim={self.aaindex_dim}"
            )
        if self.fusion_type not in {"gated_add", "concat", "add", "swiglu"}:
            raise ValueError(
                "fusion_type must be one of gated_add, concat, add, swiglu, "
                f"got {self.fusion_type!r}"
            )
        if self.post_sequence_model not in {"none", "bigru", "lstm", "gru", "transformer", "cnn", "attention"}:
            raise ValueError(
                "post_sequence_model must be one of none, bigru, lstm, gru, transformer, cnn, attention, "
                f"got {self.post_sequence_model!r}"
            )
        if not self.use_post_fusion_bigru and self.post_sequence_model == "bigru":
            self.post_sequence_model = "none"
        self.use_post_fusion_bigru = self.post_sequence_model == "bigru"
        if self.num_cnn_layers < 0:
            raise ValueError(f"num_cnn_layers must be non-negative, got {self.num_cnn_layers}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.d % self.num_heads != 0:
            raise ValueError(f"d must be divisible by num_heads, got d={self.d}, num_heads={self.num_heads}")
        has_esm_branch = not self.disable_esm
        has_p0_branch = not self.disable_p0 and self.p0_dim > 0
        has_local_branch = not self.disable_token_cnn
        if not (has_esm_branch or has_p0_branch or has_local_branch):
            raise ValueError("At least one branch must remain enabled: ESM, P0, or token-CNN.")
        if self.use_prior_bottleneck and self.prior_bottleneck_dim <= 0:
            raise ValueError(
                "prior_bottleneck_dim must be positive when use_prior_bottleneck=True, "
                f"got {self.prior_bottleneck_dim}"
            )
        if self.cross_attention_direction not in {"prior_to_semantic", "semantic_to_prior"}:
            raise ValueError(
                "cross_attention_direction must be 'prior_to_semantic' or 'semantic_to_prior', "
                f"got {self.cross_attention_direction!r}"
            )
        if self.h_embed_update not in {"start_nhl_epoch_once", "every_epoch", "every_n_epochs"}:
            raise ValueError(
                "h_embed_update must be start_nhl_epoch_once, every_epoch, or every_n_epochs, "
                f"got {self.h_embed_update!r}"
            )
        if not 0.0 <= self.h_embed_ema_decay < 1.0:
            raise ValueError(
                f"h_embed_ema_decay must be in [0, 1), got {self.h_embed_ema_decay}"
            )
        if self.h_embed_update_interval < 1:
            raise ValueError(
                f"h_embed_update_interval must be >= 1, got {self.h_embed_update_interval}"
            )
        if self.threshold_objective != "MCC":
            raise ValueError(f"Only threshold_objective='MCC' is currently supported, got {self.threshold_objective!r}")
        if isinstance(self.swanlab_tags, str):
            self.swanlab_tags = _parse_str_list(self.swanlab_tags)
        elif self.swanlab_tags is None:
            self.swanlab_tags = []
        else:
            self.swanlab_tags = [str(tag) for tag in self.swanlab_tags if str(tag).strip()]

    def apply_data_root(self, data_root: str | Path, *, override_train: bool, override_test: bool) -> None:
        root = _path_from(data_root, Path(self.project_root))
        if override_train:
            self.train_tsv_path = str(root / DEFAULT_TRAIN_TSV_NAME)
        if override_test:
            self.test_tsv_path = str(root / DEFAULT_TEST_TSV_NAME)

    def apply_environment_overrides(self) -> None:
        project_root = os.environ.get("GLACE_PROJECT_ROOT")
        if project_root:
            self.project_root = _project_root_string(project_root)
        data_root = os.environ.get("GLACE_DATA_ROOT")
        if data_root:
            self.apply_data_root(
                data_root,
                override_train=not os.environ.get("GLACE_TRAIN_TSV"),
                override_test=not os.environ.get("GLACE_TEST_TSV"),
            )
        for attr, env_name in PATH_ENV_VARS.items():
            if attr == "project_root":
                continue
            value = os.environ.get(env_name)
            if value:
                if attr == "esm_model_dir":
                    registry_name = _normalise_esm_registry_name(value)
                    if registry_name is not None:
                        setattr(self, attr, registry_name)
                        continue
                setattr(self, attr, _path_string(value, Path(self.project_root)))
        api_key = os.environ.get("SWANLAB_API_KEY")
        if api_key and not self.swanlab_api_key:
            self.swanlab_api_key = api_key

    def rebase_missing_project_paths(self, saved_project_root: str | None) -> None:
        if not saved_project_root:
            return
        current_root = _configured_project_root()
        for attr in PATH_FIELDS:
            value = getattr(self, attr)
            if not value or Path(value).exists():
                continue
            rebased = _rebase_path(str(value), str(saved_project_root), current_root)
            if rebased is None:
                saved_parent = str(PureWindowsPath(str(saved_project_root)).parent)
                rebased = _rebase_path(str(value), saved_parent, current_root.parent)
            if rebased is not None:
                setattr(self, attr, rebased)
        self.project_root = str(current_root)


# =============================================================================
# 10. 命令行参数 / 工厂函数
# =============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or run MICA-ACP.")
    # 路径
    parser.add_argument("--project_root")
    parser.add_argument("--data_root")
    parser.add_argument("--train_tsv", dest="train_tsv_path")
    parser.add_argument("--valid_tsv", dest="valid_tsv_path")
    parser.add_argument("--test_tsv", dest="test_tsv_path")
    parser.add_argument("--aaindex_csv", dest="aaindex_csv_path")
    parser.add_argument("--blosum_csv", dest="blosum_csv_path")
    parser.add_argument("--esm_model_dir")
    parser.add_argument("--esm_model_name")
    parser.add_argument("--esm_dim", type=int)
    parser.add_argument("--cache_dir")
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--log_dir")
    parser.add_argument("--Lmax", type=int, help="最大序列长度；影响截断、padding、P0 和 ESM cache。")
    parser.add_argument(
        "--truncate_strategy",
        choices=["head", "tail", "center", "head_tail"],
        help="超长序列截断策略。",
    )
    # 消融实验
    parser.add_argument("--aaindex_indices", help="自定义 AAindex 行索引,逗号分隔,如 109,268,396")
    parser.add_argument("--aaindex_dim", type=int, help="使用前 N 个 AAindex 索引;0 表示只用 BLOSUM。")
    parser.add_argument("--select_aaindex_mrmr", action="store_true", help="训练开始前用训练集做稳定 mRMR 筛选 AAindex。")
    parser.add_argument("--aaindex_selection_repeats", type=int)
    parser.add_argument("--aaindex_selection_sample_fraction", type=float)
    parser.add_argument("--aaindex_redundancy_threshold", type=float)
    parser.add_argument("--no_aac20_prior", dest="use_aac20_prior", action="store_false", default=None)
    parser.add_argument("--disable_blosum", action="store_true", help="关闭 BLOSUM 先验。")
    parser.add_argument("--disable_esm", action="store_true", help="关闭 ESM-C 语义分支。")
    parser.add_argument("--disable_p0", action="store_true", help="关闭 P0 理化先验分支。")
    parser.add_argument("--disable_token_cnn", action="store_true", help="关闭 token embedding/CNN 局部分支。")
    parser.add_argument("--disable_cnn", dest="disable_token_cnn", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cnn_kernels", help="奇数卷积核列表,逗号分隔,如 3,5,7")
    parser.add_argument("--num_cnn_layers", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--d", type=int, dest="d", help="主干隐藏维度 d")
    parser.add_argument("--d_ff", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument(
        "--use_prior_bottleneck",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否在 P0 分支启用中间瓶颈层；默认 False（单层 p0->d 投影）。",
    )
    parser.add_argument("--prior_bottleneck_dim", type=int, help="启用瓶颈时的中间维度，需 >0。")
    parser.add_argument("--prior_dropout", type=float)
    parser.add_argument("--scale_dropout", type=float)
    parser.add_argument(
        "--cross_attention_direction",
        choices=["prior_to_semantic", "semantic_to_prior"],
        help="prior_to_semantic 为 CA-old,semantic_to_prior 为 CA-new。",
    )
    parser.add_argument("--disable_cross_attention", action="store_true", help="关闭 P0-ESM cross-attention。")
    parser.add_argument("--branch_dropout", type=float)
    parser.add_argument(
        "--fusion_type",
        choices=["gated_add", "concat", "add", "swiglu"],
        help="semantic/local 融合方式。",
    )
    parser.add_argument(
        "--post_sequence_model",
        choices=["none", "bigru", "lstm", "gru", "transformer", "cnn", "attention"],
        help="融合后的序列建模模块。",
    )
    parser.add_argument("--use_post_fusion_bigru", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--bigru_hidden", type=int)
    parser.add_argument("--bigru_layers", type=int)
    parser.add_argument("--bigru_dropout", type=float)
    # 训练
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", dest="resume_path")
    parser.add_argument("--force_rebuild_cache", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument(
        "--use_test_as_valid",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否把 test 集当验证集（选 best ckpt / 阈值）；默认 True，使用 --no-use_test_as_valid 关闭并改用 valid_ratio 切分。",
    )
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--use_scheduler", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--lr_scheduler_monitor")
    parser.add_argument("--lr_scheduler_mode", choices=["min", "max"])
    parser.add_argument("--early_stopping_patience", type=int)
    parser.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ema_decay", type=float)
    parser.add_argument("--ema_start_epoch", type=int)
    parser.add_argument("--use_rdrop", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--lambda_rdrop", type=float)
    parser.add_argument("--lambda_hn", type=float)
    parser.add_argument("--topk_pos", type=int)
    parser.add_argument("--rho", type=float)
    parser.add_argument("--tau_loss", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--start_nhl_epoch", type=int)
    parser.add_argument("--nhl_warmup_epochs", type=int)
    parser.add_argument("--h_embed_update", choices=["start_nhl_epoch_once", "every_epoch", "every_n_epochs"])
    parser.add_argument("--h_embed_update_interval", type=int)
    parser.add_argument(
        "--h_embed_ema_decay",
        type=float,
        help="H_embed 动量混合系数 ∈ [0,1)。0=直接覆盖（旧行为）；推荐 0.8~0.95，越大越平滑。",
    )
    parser.add_argument("--w_hn_max", type=float)
    parser.add_argument("--threshold_objective", choices=["MCC"])
    parser.add_argument("--threshold_min_specificity", type=float)
    parser.add_argument("--threshold_search_min", type=float)
    parser.add_argument("--threshold_search_max", type=float)
    parser.add_argument("--threshold_search_step", type=float)
    # 实验隔离
    parser.add_argument(
        "--run_name",
        help="本次实验的子目录名;不传则按关键超参 + 时间戳自动生成。",
    )
    # SwanLab
    parser.add_argument(
        "--use_swanlab",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否启用 SwanLab 实验追踪；默认 True，使用 --no-use_swanlab 关闭。",
    )
    parser.add_argument("--swanlab_project", help="SwanLab 项目名 (默认 MICA-ACP)。")
    parser.add_argument("--swanlab_experiment", help="SwanLab 实验/run 名。")
    parser.add_argument("--swanlab_workspace", help="SwanLab workspace / 组织。")
    parser.add_argument(
        "--swanlab_mode",
        choices=["cloud", "local", "offline", "disabled"],
        help="SwanLab mode 传给 swanlab.init。",
    )
    parser.add_argument("--swanlab_api_key", help="SwanLab API key (覆盖环境变量 SWANLAB_API_KEY)。")
    parser.add_argument("--swanlab_description", help="SwanLab run 的描述文本。")
    parser.add_argument("--swanlab_tags", help="逗号分隔的 SwanLab 标签,如 baseline,esmc600m。")
    parser.add_argument("--swanlab_logdir", help="SwanLab 本地日志目录。")
    return parser


def config_from_args(args: argparse.Namespace | None = None) -> MICAConfig:
    cfg = MICAConfig()
    cfg.apply_environment_overrides()
    if args is None:
        args = build_arg_parser().parse_args()
    custom_aaindex_dim = getattr(args, "aaindex_dim", None) is not None
    if getattr(args, "project_root", None) is not None:
        cfg.project_root = _project_root_string(args.project_root)
    data_root = getattr(args, "data_root", None)
    if data_root is not None:
        cfg.apply_data_root(
            data_root,
            override_train=getattr(args, "train_tsv_path", None) is None,
            override_test=getattr(args, "test_tsv_path", None) is None,
        )
    for key, value in vars(args).items():
        if key == "data_root":
            continue
        if value is not None and hasattr(cfg, key):
            if key == "project_root":
                value = _project_root_string(value)
            elif key in PATH_FIELDS:
                if key == "esm_model_dir":
                    registry_name = _normalise_esm_registry_name(value)
                    value = registry_name if registry_name is not None else _path_string(value, Path(cfg.project_root))
                else:
                    value = _path_string(value, Path(cfg.project_root))
            elif key == "aaindex_indices":
                value = _parse_int_list(value)
                if not custom_aaindex_dim:
                    cfg.aaindex_dim = len(value)
            elif key == "cnn_kernels":
                value = _parse_int_list(value)
            elif key == "swanlab_tags":
                value = _parse_str_list(value)
            setattr(cfg, key, value)
    if getattr(args, "force_rebuild_cache", False):
        cfg.force_rebuild_cache = True
    if getattr(args, "use_amp", False):
        cfg.use_amp = True
    cfg.sync_derived_dims()
    return cfg


def get_config() -> MICAConfig:
    cfg = MICAConfig()
    cfg.apply_environment_overrides()
    cfg.sync_derived_dims()
    return cfg


def get_train_config() -> MICAConfig:
    return get_config()
