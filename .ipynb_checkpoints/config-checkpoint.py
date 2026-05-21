
from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_TRAIN_TSV_NAME = "ACP2_main_train.tsv"
DEFAULT_TEST_TSV_NAME = "ACP2_main_test.tsv"

ESM_MODEL_REGISTRY_NAMES = {"esmc_300m", "esmc_600m"}


AAINDEX_TOTAL_ROWS: int = 553

# 全 553 维候选 (无任何筛选,按 AAINDEX.csv 原始行序)
AAINDEX_FULL_INDICES: list[int] = list(range(AAINDEX_TOTAL_ROWS))

# 按重要度排序的 60 列子集 (做小规模消融时使用)
AAINDEX_TOP60_INDICES: list[int] = [
    109, 268, 396,  92, 282,  14, 269, 220,  58, 307,
    332,  56, 327, 405, 415,  84, 472, 238, 485,  86,
    299, 187, 433, 292, 443, 266,  64, 103, 280, 340,
     48, 106, 257, 481,  91, 400, 267, 449, 217, 399,
    499,  54, 445,  23, 391, 314, 218,  40, 101, 291,
    473, 422, 226, 265, 442,  36, 414, 154, 426, 172,
]

# >>>>> 选择本轮使用的索引池 + 维度 (做消融实验主要修改这两行) <<<<<
# 用法 1 (当前): 全部 553 维
#     DEFAULT_AAINDEX_INDICES = AAINDEX_FULL_INDICES
#     DEFAULT_AAINDEX_DIM     = AAINDEX_TOTAL_ROWS    # = 553
# 用法 2: 重要度 Top60 子集 (取前 N 个,N ∈ {0, 10, 20, 30, 40, 50, 60})
#     DEFAULT_AAINDEX_INDICES = AAINDEX_TOP60_INDICES
#     DEFAULT_AAINDEX_DIM     = 60
# 用法 3: 关闭 AAINDEX 通道 (只用 BLOSUM62)
#     DEFAULT_AAINDEX_INDICES = AAINDEX_FULL_INDICES   # 任意池都行
#     DEFAULT_AAINDEX_DIM     = 0
DEFAULT_AAINDEX_INDICES: list[int] = AAINDEX_FULL_INDICES
DEFAULT_AAINDEX_DIM: int = AAINDEX_TOTAL_ROWS

# --- CNN 多尺度卷积核 --------------------------------------------------------
# 必须为奇数列表,长度 = 卷积分支数。
# 推荐做消融的取值: [3], [3,5], [3,5,7], [5,7,9] 等。
DEFAULT_CNN_KERNELS: list[int] = [3, 5,7]

# CNN 堆叠层数 (每层包含 1 个 Conv + GeLU + Norm)
DEFAULT_NUM_CNN_LAYERS: int = 2

# --- Transformer / 注意力 ----------------------------------------------------
DEFAULT_NUM_HEADS: int = 8           # Multi-head attention 头数
DEFAULT_HIDDEN_DIM: int = 256        # 主干隐藏维度 d
DEFAULT_FFN_DIM: int = 512           # FFN (SwiGLU) 中间维度

# --- 正则化 -----------------------------------------------------------------
DEFAULT_DROPOUT: float = 0.3


# =============================================================================
# 3. 输入特征维度 (一般不修改)
# =============================================================================
DEFAULT_LMAX: int = 50               # 序列最大长度
DEFAULT_BLOSUM_DIM: int = 20         # BLOSUM62 维度 (固定)
DEFAULT_ESM_DIM: int = 1152          # ESM-C-600M 输出维度
DEFAULT_ESM_MODEL_NAME: str = "ESM-C-600M"


# =============================================================================
# 4. 训练超参
# =============================================================================
DEFAULT_EPOCHS: int = 200
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_LR: float = 1e-4
DEFAULT_WEIGHT_DECAY: float = 1e-2
DEFAULT_CLIP_GRAD_NORM: float = 1.0
DEFAULT_EARLY_STOPPING_PATIENCE: int = 20
DEFAULT_SEED: int = 42
DEFAULT_VALID_RATIO: float = 0.2
DEFAULT_THRESHOLD: float = 0.5
DEFAULT_NUM_WORKERS: int = 2
DEFAULT_USE_AMP: bool = False
DEFAULT_OPTIMIZER: str = "AdamW"
DEFAULT_LR_SCHEDULER_MONITOR: str = "valid_loss"
DEFAULT_LR_SCHEDULER_MODE: str = "min"


# =============================================================================
# 5. Hard Negative Mining 参数
# =============================================================================
DEFAULT_TOPK_POS: int = 5
DEFAULT_RHO: float = 0.9
DEFAULT_TAU_LOSS: float = 0.95
DEFAULT_ALPHA: float = 0.6
DEFAULT_BETA: float = 0.5
DEFAULT_LAMBDA_HN: float = 1.0
DEFAULT_E_HN_WARM: int = 10
DEFAULT_GATE_INIT: float = -2.2


# =============================================================================
# 6. SwanLab 实验追踪
# =============================================================================
DEFAULT_USE_SWANLAB: bool = False
DEFAULT_SWANLAB_PROJECT: str = "MICA-ACP"
DEFAULT_SWANLAB_API_KEY: str | None = "pWSSR0qR01AUw2IeRUteN"


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
    project_root: str = field(default_factory=lambda: str(_configured_project_root()))
    train_tsv_path: str = field(default_factory=_default_train_tsv_path)
    valid_tsv_path: str | None = field(default_factory=_default_valid_tsv_path)
    test_tsv_path: str = field(default_factory=_default_test_tsv_path)
    aaindex_csv_path: str = field(default_factory=_default_aaindex_path)
    blosum_csv_path: str = field(default_factory=_default_blosum_path)
    esm_model_dir: str = field(default_factory=_default_esm_model_dir)
    cache_dir: str = field(default_factory=lambda: str(_configured_project_root() / "cache"))
    checkpoint_dir: str = field(default_factory=lambda: str(_configured_project_root() / "checkpoints"))
    log_dir: str = field(default_factory=lambda: str(_configured_project_root() / "logs"))

    # ---- 输入特征维度 ----
    Lmax: int = DEFAULT_LMAX
    blosum_dim: int = DEFAULT_BLOSUM_DIM
    esm_dim: int = DEFAULT_ESM_DIM
    esm_model_name: str = DEFAULT_ESM_MODEL_NAME

    # ---- 消融:AAINDEX ----
    aaindex_dim: int = DEFAULT_AAINDEX_DIM
    aaindex_indices: list[int] = field(default_factory=lambda: list(DEFAULT_AAINDEX_INDICES))
    p0_dim: int = DEFAULT_BLOSUM_DIM + DEFAULT_AAINDEX_DIM  # 自动同步,无需手动改

    # ---- 消融:架构 ----
    d: int = DEFAULT_HIDDEN_DIM
    num_heads: int = DEFAULT_NUM_HEADS
    cnn_kernels: list[int] = field(default_factory=lambda: list(DEFAULT_CNN_KERNELS))
    num_cnn_layers: int = DEFAULT_NUM_CNN_LAYERS
    d_ff: int = DEFAULT_FFN_DIM
    dropout: float = DEFAULT_DROPOUT
    gate_init: float = DEFAULT_GATE_INIT

    # ---- Hard Negative Mining ----
    topk_pos: int = DEFAULT_TOPK_POS
    rho: float = DEFAULT_RHO
    tau_loss: float = DEFAULT_TAU_LOSS
    alpha: float = DEFAULT_ALPHA
    beta: float = DEFAULT_BETA
    lambda_hn: float = DEFAULT_LAMBDA_HN
    E_HN_warm: int = DEFAULT_E_HN_WARM

    # ---- 训练 ----
    optimizer: str = DEFAULT_OPTIMIZER
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    clip_grad_norm: float = DEFAULT_CLIP_GRAD_NORM
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    seed: int = DEFAULT_SEED
    valid_ratio: float = DEFAULT_VALID_RATIO
    positive_label_value: int = 1
    threshold: float = DEFAULT_THRESHOLD
    num_workers: int = DEFAULT_NUM_WORKERS
    pin_memory: bool = True
    use_amp: bool = DEFAULT_USE_AMP
    resume_path: str | None = None
    force_rebuild_cache: bool = False
    use_test_as_valid: bool = False
    lr_scheduler_monitor: str = DEFAULT_LR_SCHEDULER_MONITOR
    lr_scheduler_mode: str = DEFAULT_LR_SCHEDULER_MODE

    # ---- 实验隔离 ----
    # 当 run_name 设置后,checkpoint_dir/log_dir 会被自动包到同名子目录里,
    # 不同消融实验互不覆盖。留空则在 train.py 启动时根据关键超参 + 时间戳自动生成。
    run_name: str | None = None

    # ---- SwanLab ----
    use_swanlab: bool = DEFAULT_USE_SWANLAB
    swanlab_project: str = DEFAULT_SWANLAB_PROJECT
    swanlab_experiment: str | None = None
    swanlab_workspace: str | None = None
    swanlab_mode: str | None = None
    swanlab_api_key: str | None = DEFAULT_SWANLAB_API_KEY
    swanlab_description: str | None = None
    swanlab_tags: list[str] = field(default_factory=list)
    swanlab_logdir: str | None = None

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
            f"k{kernels}",
            f"L{self.num_cnn_layers}",
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
        self.aaindex_indices = _parse_int_list(self.aaindex_indices)
        if self.aaindex_dim < 0:
            raise ValueError(f"aaindex_dim must be non-negative, got {self.aaindex_dim}")
        if self.aaindex_dim > len(self.aaindex_indices):
            raise ValueError(
                "aaindex_dim cannot exceed the number of configured aaindex_indices: "
                f"aaindex_dim={self.aaindex_dim}, len(indices)={len(self.aaindex_indices)}"
            )
        self.aaindex_indices = self.aaindex_indices[: self.aaindex_dim]
        self.p0_dim = self.blosum_dim + self.aaindex_dim
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
    # 消融实验
    parser.add_argument("--aaindex_indices", help="自定义 AAindex 行索引,逗号分隔,如 109,268,396")
    parser.add_argument("--aaindex_dim", type=int, help="使用前 N 个 AAindex 索引;0 表示只用 BLOSUM。")
    parser.add_argument("--cnn_kernels", help="奇数卷积核列表,逗号分隔,如 3,5,7")
    parser.add_argument("--num_cnn_layers", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--d", type=int, dest="d", help="主干隐藏维度 d")
    parser.add_argument("--d_ff", type=int)
    parser.add_argument("--dropout", type=float)
    # 训练
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", dest="resume_path")
    parser.add_argument("--force_rebuild_cache", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--use_test_as_valid", action="store_true")
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--lr_scheduler_monitor")
    parser.add_argument("--lr_scheduler_mode", choices=["min", "max"])
    parser.add_argument("--early_stopping_patience", type=int)
    # 实验隔离
    parser.add_argument(
        "--run_name",
        help="本次实验的子目录名;不传则按关键超参 + 时间戳自动生成。",
    )
    # SwanLab
    parser.add_argument("--use_swanlab", action="store_true", help="启用 SwanLab 实验追踪。")
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
    if getattr(args, "use_test_as_valid", False):
        cfg.use_test_as_valid = True
    if getattr(args, "use_swanlab", False):
        cfg.use_swanlab = True
    cfg.sync_derived_dims()
    return cfg


def get_config() -> MICAConfig:
    cfg = MICAConfig()
    cfg.apply_environment_overrides()
    cfg.sync_derived_dims()
    return cfg


def get_train_config() -> MICAConfig:
    return get_config()
