from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import torch

from config import MICAConfig
from dataprocess.dataset import PeptideSample


CACHE_VERSION = 4
ESMC_MODEL_SPECS = {
    "esmc_300m": {"d_model": 960, "n_heads": 15, "n_layers": 30},
    "esmc_600m": {"d_model": 1152, "n_heads": 18, "n_layers": 36},
}


def _normalise_esmc_model_key(value: str | Path | None) -> str | None:
    if value is None:
        return None
    text = str(value).lower().replace("-", "_")
    if "300m" in text or "300_m" in text:
        return "esmc_300m"
    if "600m" in text or "600_m" in text:
        return "esmc_600m"
    return None


def _normalise_esmc_registry_key(value: str | Path | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_")
    return text if text in ESMC_MODEL_SPECS else None


def _looks_like_transformers_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "special_tokens_map.json",
    )
    return (path / "config.json").exists() and any((path / name).exists() for name in tokenizer_files)


def _cache_digest(samples: Sequence[PeptideSample], cfg: MICAConfig, split_name: str) -> str:
    hasher = hashlib.sha1()
    hasher.update(str(CACHE_VERSION).encode("utf-8"))
    hasher.update(split_name.encode("utf-8"))
    hasher.update(str(cfg.Lmax).encode("utf-8"))
    hasher.update(str(cfg.esm_dim).encode("utf-8"))
    hasher.update(str(cfg.esm_model_name).encode("utf-8"))
    for sample in samples:
        hasher.update(str(sample.sample_index).encode("utf-8"))
        hasher.update(sample.sequence.encode("utf-8"))
        hasher.update(str(sample.label).encode("utf-8"))
    return hasher.hexdigest()[:16]


def cache_file_for_split(samples: Sequence[PeptideSample], cfg: MICAConfig, split_name: str) -> Path:
    digest = _cache_digest(samples, cfg, split_name)
    return Path(cfg.cache_dir) / f"{split_name}_esmc_l{cfg.Lmax}_d{cfg.esm_dim}_{digest}.pt"


class ESMEmbeddingExtractor:
    def __init__(self, cfg: MICAConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.backend = ""
        self.model = None
        self.tokenizer = None
        self.logits_config_cls = None
        self.protein_cls = None
        self._load_model()

    def _load_model(self) -> None:
        model_ref = str(self.cfg.esm_model_dir).strip()
        model_path = Path(model_ref).expanduser()
        model_key = _normalise_esmc_model_key(self.cfg.esm_model_name) or _normalise_esmc_model_key(model_ref)
        registry_model_key = _normalise_esmc_registry_key(model_ref)

        errors: list[str] = []

        if model_path.exists():
            if _looks_like_transformers_dir(model_path):
                try:
                    self._load_transformers_model(model_path)
                except Exception as exc:  # pragma: no cover - depends on local model package
                    errors.append(f"transformers backend failed: {exc}")

            if self.model is None:
                try:
                    self._load_local_pth_model(model_path, model_key)
                except Exception as exc:  # pragma: no cover - depends on local model package
                    errors.append(f"esm local .pth backend failed: {exc}")
        elif registry_model_key is not None:
            try:
                self._load_registered_esmc_model(registry_model_key)
            except Exception as exc:  # pragma: no cover - depends on local model package
                errors.append(f"esm registered from_pretrained backend failed: {exc}")
        else:
            raise FileNotFoundError(
                "ESM-C model source does not exist and is not a known ESM-C registry name. "
                f"Tried: {model_ref}. Use a local .pth file/directory or one of: "
                f"{', '.join(sorted(ESMC_MODEL_SPECS))}."
            )

        if self.model is None:
            detail = "\n".join(errors)
            raise RuntimeError(
                f"Failed to load ESM-C from {model_ref}.\n"
                "Install a compatible `esm` package and ensure the local .pth weights are complete.\n"
                f"{detail}"
            )

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _load_transformers_model(self, model_dir: Path) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            local_files_only=True,
        ).to(self.device)
        self.backend = "transformers"

    def _load_local_pth_model(self, model_path: Path, model_key: str | None) -> None:
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.tokenization import get_esmc_model_tokenizers

        weight_path = self._find_weight_file(model_path)
        model_key = model_key or _normalise_esmc_model_key(weight_path)
        if model_key not in ESMC_MODEL_SPECS:
            raise RuntimeError(
                "Cannot infer ESM-C model size from the weight filename or config. "
                "Use a filename containing 300m/600m or set cfg.esm_model_name accordingly."
            )
        spec = ESMC_MODEL_SPECS[model_key]
        if spec["d_model"] != self.cfg.esm_dim:
            raise RuntimeError(
                f"{model_key} outputs {spec['d_model']} dims, but cfg.esm_dim={self.cfg.esm_dim}. "
                "Use the matching ESM-C model size or update esm_dim before training."
            )

        self.model = ESMC(
            d_model=spec["d_model"],
            n_heads=spec["n_heads"],
            n_layers=spec["n_layers"],
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=False,
        ).to(self.device)
        self.model.load_state_dict(self._load_state_dict(weight_path))
        self.protein_cls = ESMProtein
        self.logits_config_cls = LogitsConfig
        self.backend = "esm"

    def _load_registered_esmc_model(self, model_key: str) -> None:
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig

        spec = ESMC_MODEL_SPECS[model_key]
        if spec["d_model"] != self.cfg.esm_dim:
            raise RuntimeError(
                f"{model_key} outputs {spec['d_model']} dims, but cfg.esm_dim={self.cfg.esm_dim}. "
                "Use the matching ESM-C model size or update esm_dim before training."
            )
        self.model = ESMC.from_pretrained(model_key, device=self.device).to(self.device)
        self.protein_cls = ESMProtein
        self.logits_config_cls = LogitsConfig
        self.backend = "esm"

    def _find_weight_file(self, model_path: Path) -> Path:
        if model_path.is_file():
            if model_path.suffix != ".pth":
                raise FileNotFoundError(f"ESM-C local weight file must be .pth, got: {model_path}")
            return model_path
        pth_files = sorted(model_path.glob("*.pth"))
        if not pth_files:
            pth_files = sorted(model_path.glob("**/*.pth"))
        if not pth_files:
            raise FileNotFoundError(f"No .pth weight file found under {model_path}")
        return pth_files[0]

    def _load_state_dict(self, weight_path: Path) -> dict[str, torch.Tensor]:
        try:
            payload = torch.load(weight_path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(weight_path, map_location=self.device)
        if isinstance(payload, dict) and "state_dict" in payload:
            payload = payload["state_dict"]
        if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
            payload = payload["model"]
        if not isinstance(payload, dict):
            raise RuntimeError(f"ESM-C weight file did not contain a state_dict: {weight_path}")
        return {str(key).removeprefix("module."): value for key, value in payload.items()}

    @torch.no_grad()
    def extract_one(self, sequence: str) -> torch.Tensor:
        if self.backend == "transformers":
            return self._extract_transformers(sequence)
        if self.backend == "esm":
            return self._extract_esm_sdk(sequence)
        raise RuntimeError("ESMEmbeddingExtractor has no active backend.")

    @torch.no_grad()
    def _extract_transformers(self, sequence: str) -> torch.Tensor:
        assert self.tokenizer is not None and self.model is not None
        inputs = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        if not hasattr(outputs, "last_hidden_state"):
            raise RuntimeError("Transformers ESM model output does not expose last_hidden_state.")
        hidden = outputs.last_hidden_state.squeeze(0).detach().cpu()
        return self._align_and_pad(hidden, len(sequence))

    @torch.no_grad()
    def _extract_esm_sdk(self, sequence: str) -> torch.Tensor:
        assert self.model is not None and self.protein_cls is not None and self.logits_config_cls is not None
        protein = self.protein_cls(sequence=sequence)
        protein_tensor = self.model.encode(protein)
        if hasattr(protein_tensor, "to"):
            protein_tensor = protein_tensor.to(self.device)
        logits_config = self.logits_config_cls(sequence=True, return_embeddings=True)
        output = self.model.logits(protein_tensor, logits_config)
        if not hasattr(output, "embeddings"):
            raise RuntimeError("ESM SDK output does not expose embeddings.")
        hidden = output.embeddings
        if hidden.dim() == 3:
            hidden = hidden.squeeze(0)
        hidden = hidden.detach().cpu()
        return self._align_and_pad(hidden, len(sequence))

    def _align_and_pad(self, hidden: torch.Tensor, valid_len: int) -> torch.Tensor:
        if hidden.dim() != 2:
            raise RuntimeError(f"ESM-C residue embedding must be rank 2, got {tuple(hidden.shape)}")
        if hidden.shape[-1] != self.cfg.esm_dim:
            raise RuntimeError(
                f"ESM-C embedding dim mismatch: expected {self.cfg.esm_dim}, got {hidden.shape[-1]}"
            )
        if hidden.shape[0] == valid_len:
            residues = hidden
        elif hidden.shape[0] >= valid_len + 2:
            residues = hidden[1 : 1 + valid_len]
        elif hidden.shape[0] > valid_len:
            residues = hidden[:valid_len]
        else:
            raise RuntimeError(
                "ESM-C output cannot be aligned to residues: "
                f"sequence length={valid_len}, output length={hidden.shape[0]}"
            )
        out = torch.zeros(self.cfg.Lmax, self.cfg.esm_dim, dtype=torch.float32)
        out[:valid_len] = residues[:valid_len].float()
        return out


def _cache_metadata_matches(payload: dict, cfg: MICAConfig) -> bool:
    return (
        payload.get("version") == CACHE_VERSION
        and payload.get("Lmax") == cfg.Lmax
        and payload.get("esm_dim") == cfg.esm_dim
        and payload.get("esm_model_name") == cfg.esm_model_name
    )


def _load_cache(
    path: Path,
    cfg: MICAConfig,
    samples: Sequence[PeptideSample],
) -> dict[int, torch.Tensor] | None:
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Failed to load ESM cache {path}: {exc}") from exc
    if not isinstance(payload, dict) or "records" not in payload:
        raise RuntimeError(f"ESM cache is malformed: {path}")
    if not _cache_metadata_matches(payload, cfg):
        return None
    expected = {int(sample.sample_index): sample for sample in samples}
    features: dict[int, torch.Tensor] = {}
    for record in payload["records"]:
        for key in ("sequence", "sample_index", "H", "mask", "label"):
            if key not in record:
                raise RuntimeError(f"ESM cache record is missing `{key}` in {path}")
        sample_index = int(record["sample_index"])
        if sample_index not in expected:
            return None
        sample = expected[sample_index]
        if record["sequence"] != sample.sequence or float(record["label"]) != float(sample.label):
            return None
        h = torch.as_tensor(record["H"], dtype=torch.float32)
        mask = torch.as_tensor(record["mask"], dtype=torch.float32)
        if h.shape != (cfg.Lmax, cfg.esm_dim):
            raise RuntimeError(f"ESM cache H shape mismatch in {path}: got {tuple(h.shape)}")
        if mask.shape != (cfg.Lmax,):
            raise RuntimeError(f"ESM cache mask shape mismatch in {path}: got {tuple(mask.shape)}")
        expected_mask = torch.zeros(cfg.Lmax, dtype=torch.float32)
        expected_mask[: len(sample.sequence)] = 1.0
        if not torch.equal(mask, expected_mask):
            raise RuntimeError(f"ESM cache mask does not match sequence length in {path}")
        padding_values = h[mask == 0]
        if padding_values.numel() and not torch.allclose(padding_values, torch.zeros_like(padding_values)):
            raise RuntimeError(f"ESM cache has non-zero padding embeddings: {path}")
        features[sample_index] = h
    if set(features) != set(expected):
        return None
    return features


def ensure_esm_embeddings(
    samples: Sequence[PeptideSample],
    cfg: MICAConfig,
    split_name: str,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    if cfg.disable_esm:
        return {
            int(sample.sample_index): torch.zeros(cfg.Lmax, cfg.esm_dim, dtype=torch.float32)
            for sample in samples
        }
    Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
    cache_path = cache_file_for_split(samples, cfg, split_name)
    if cache_path.exists() and not cfg.force_rebuild_cache:
        cached = _load_cache(cache_path, cfg, samples)
        if cached is not None:
            return cached

    extractor = ESMEmbeddingExtractor(cfg, device)
    records = []
    features: dict[int, torch.Tensor] = {}
    for sample in samples:
        h = extractor.extract_one(sample.sequence)
        mask = torch.zeros(cfg.Lmax, dtype=torch.float32)
        mask[: len(sample.sequence)] = 1.0
        assert h.shape == (cfg.Lmax, cfg.esm_dim)
        assert mask.shape == (cfg.Lmax,)
        features[sample.sample_index] = h
        records.append(
            {
                "sequence": sample.sequence,
                "sample_index": int(sample.sample_index),
                "H": h,
                "mask": mask,
                "label": float(sample.label),
            }
        )
    torch.save(
        {
            "version": CACHE_VERSION,
            "Lmax": cfg.Lmax,
            "esm_dim": cfg.esm_dim,
            "esm_model_name": cfg.esm_model_name,
            "records": records,
        },
        cache_path,
    )
    return features
