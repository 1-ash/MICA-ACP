"""SwanLab logging helpers for MICA-ACP training.

This module wraps the optional ``swanlab`` dependency so the training
pipeline can run with or without SwanLab installed. When SwanLab is not
installed, or when ``cfg.use_swanlab`` is False, every helper turns into a
no-op so the rest of the codebase does not need to scatter ``if`` checks.

Typical usage:

    logger = SwanLabLogger.from_config(cfg)
    logger.log_train_epoch(epoch, train_metrics, lr=lr)
    logger.log_valid_epoch(epoch, valid_metrics)
    logger.log_test_summary(test_metrics)
    logger.finish()

Set ``cfg.use_swanlab=True`` (or pass ``--use_swanlab``) to enable. The
SwanLab project / experiment / workspace / mode / api_key fields on the
config map directly onto ``swanlab.init`` arguments. When ``cfg.use_swanlab``
is False or SwanLab is missing, ``SwanLabLogger.from_config`` returns a
stub instance whose methods are no-ops.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

try:
    import swanlab  # type: ignore[import-not-found]

    _SWANLAB_AVAILABLE = True
except Exception:  # pragma: no cover - import guarded for optional dep
    swanlab = None  # type: ignore[assignment]
    _SWANLAB_AVAILABLE = False


_SCALAR_TYPES = (int, float, bool)


def _is_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return True
    if hasattr(value, "item") and callable(value.item):
        try:
            scalar = value.item()
        except Exception:
            return False
        return isinstance(scalar, _SCALAR_TYPES)
    return False


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if hasattr(value, "item") and callable(value.item):
        try:
            return float(value.item())
        except Exception:
            return None
    return None


def _flatten_metric_dict(
    metrics: Mapping[str, Any],
    prefix: str = "",
) -> dict[str, float]:
    flat: dict[str, float] = {}
    for raw_key, raw_value in metrics.items():
        key = f"{prefix}{raw_key}" if prefix else str(raw_key)
        if isinstance(raw_value, Mapping):
            flat.update(_flatten_metric_dict(raw_value, prefix=f"{key}/"))
            continue
        scalar = _as_float(raw_value)
        if scalar is None:
            continue
        flat[key] = scalar
    return flat


class SwanLabLogger:
    """Thin wrapper around the SwanLab Python SDK."""

    def __init__(
        self,
        *,
        enabled: bool,
        project: str | None = None,
        experiment_name: str | None = None,
        workspace: str | None = None,
        mode: str | None = None,
        api_key: str | None = None,
        config: Mapping[str, Any] | None = None,
        description: str | None = None,
        tags: Iterable[str] | None = None,
        logdir: str | None = None,
    ) -> None:
        self._enabled = bool(enabled and _SWANLAB_AVAILABLE)
        self._closed = False
        self._run = None
        if not self._enabled:
            return

        if api_key:
            try:
                swanlab.login(api_key=api_key, save=False)  # type: ignore[union-attr]
            except Exception as exc:  # pragma: no cover - network/login dep
                print(f"[swanlab] login failed, falling back to anonymous mode: {exc}")

        init_kwargs: dict[str, Any] = {}
        if project:
            init_kwargs["project"] = project
        if experiment_name:
            init_kwargs["experiment_name"] = experiment_name
        if workspace:
            init_kwargs["workspace"] = workspace
        if mode:
            init_kwargs["mode"] = mode
        if logdir:
            init_kwargs["logdir"] = logdir
        if description:
            init_kwargs["description"] = description
        if tags is not None:
            init_kwargs["tags"] = list(tags)
        if config is not None:
            init_kwargs["config"] = dict(config)

        try:
            self._run = swanlab.init(**init_kwargs)  # type: ignore[union-attr]
        except Exception as exc:  # pragma: no cover - depends on remote svc
            print(f"[swanlab] init failed, disabling SwanLab logging: {exc}")
            self._enabled = False
            self._run = None

    @classmethod
    def from_config(cls, cfg: Any) -> "SwanLabLogger":
        enabled = bool(getattr(cfg, "use_swanlab", False))
        if not enabled:
            return cls(enabled=False)
        if not _SWANLAB_AVAILABLE:
            print(
                "[swanlab] use_swanlab=True but the `swanlab` package is not installed; "
                "logging will be skipped. Install with `pip install swanlab` to enable."
            )
            return cls(enabled=False)

        config_payload: Mapping[str, Any] | None = None
        if hasattr(cfg, "to_dict") and callable(cfg.to_dict):
            try:
                config_payload = cfg.to_dict()
            except Exception:
                config_payload = None

        return cls(
            enabled=True,
            project=getattr(cfg, "swanlab_project", None) or "MICA-ACP",
            experiment_name=getattr(cfg, "swanlab_experiment", None),
            workspace=getattr(cfg, "swanlab_workspace", None),
            mode=getattr(cfg, "swanlab_mode", None),
            api_key=getattr(cfg, "swanlab_api_key", None),
            config=config_payload,
            description=getattr(cfg, "swanlab_description", None),
            tags=getattr(cfg, "swanlab_tags", None) or None,
            logdir=getattr(cfg, "swanlab_logdir", None),
        )

    @property
    def enabled(self) -> bool:
        return self._enabled and not self._closed

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        if not self.enabled:
            return
        flat = _flatten_metric_dict(metrics)
        if not flat:
            return
        try:
            if step is None:
                swanlab.log(flat)  # type: ignore[union-attr]
            else:
                swanlab.log(flat, step=int(step))  # type: ignore[union-attr]
        except Exception as exc:  # pragma: no cover - depends on remote svc
            print(f"[swanlab] log failed (step={step}): {exc}")

    def log_train_epoch(
        self,
        epoch: int,
        metrics: Mapping[str, Any],
        *,
        lr: float | None = None,
        gate: float | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        payload: dict[str, Any] = {"epoch": int(epoch)}
        for key, value in metrics.items():
            if not key.startswith("train"):
                payload[f"train/{key}"] = value
            else:
                short = key.removeprefix("train_") if key.startswith("train_") else key
                payload[f"train/{short}"] = value
        if lr is not None:
            payload["train/lr"] = lr
        if gate is not None:
            payload["train/gate"] = gate
        if extra:
            for key, value in extra.items():
                payload[key if "/" in key else f"train/{key}"] = value
        self.log(payload, step=int(epoch))

    def log_valid_epoch(self, epoch: int, metrics: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        payload: dict[str, Any] = {"epoch": int(epoch)}
        for key, value in metrics.items():
            payload[f"valid/{key}"] = value
        self.log(payload, step=int(epoch))

    def log_test_summary(self, metrics: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        payload = {f"test/{key}": value for key, value in metrics.items()}
        self.log(payload)
        run = self._run
        if run is None:
            return
        config_obj = getattr(run, "config", None)
        if config_obj is None:
            return
        flat = _flatten_metric_dict(metrics, prefix="test/")
        for key, value in flat.items():
            try:
                config_obj[key] = value
            except Exception:
                continue

    def log_best(self, epoch: int, metrics: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        payload: dict[str, Any] = {"best/epoch": int(epoch)}
        for key, value in metrics.items():
            payload[f"best/{key}"] = value
        self.log(payload, step=int(epoch))

    def finish(self) -> None:
        if not self.enabled:
            self._closed = True
            return
        try:
            swanlab.finish()  # type: ignore[union-attr]
        except Exception as exc:  # pragma: no cover - depends on remote svc
            print(f"[swanlab] finish failed: {exc}")
        finally:
            self._closed = True
            self._run = None

    def __enter__(self) -> "SwanLabLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish()
