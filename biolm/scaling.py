"""Helpers for building scalers + scaling specs for dataset/model code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .metrics import IdentityScaler, LogScaler


SCALER_MAPPING = {
    "minmax": MinMaxScaler,
    "min-max": MinMaxScaler,
    "standard": StandardScaler,
    "zscore": StandardScaler,
    "log": LogScaler,
    "identity": IdentityScaler,
}


@dataclass
class ScalingSpec:
    method: str
    scaler: Any


def get_scaler_for_method(method: Optional[str]) -> Any:
    """Return a scaler instance for `method` (falls back to identity)."""

    normalized = (method or "identity").strip().lower()
    scaler_cls = SCALER_MAPPING.get(normalized)
    if scaler_cls is not None:
        return scaler_cls()
    return IdentityScaler()


def resolve_scaling_method(args: Optional[Any]) -> str:
    """Determine the scaling method string from config-like args."""

    if args is None:
        return "identity"
    training = getattr(args, "training", None)
    method = None
    if training is not None:
        method = getattr(training, "scaling", None)
    if method is None:
        method = getattr(args, "scaling", None)
    return (method or "identity").strip().lower()


def build_scaling_spec(
    args: Optional[Any] = None,
    *,
    method: Optional[str] = None,
    scaler: Optional[Any] = None,
) -> ScalingSpec:
    """Construct a `ScalingSpec` using the provided overrides and config."""

    if method:
        resolved_method = method
    elif args is not None:
        resolved_method = resolve_scaling_method(args)
    else:
        resolved_method = "identity"

    resolved_method = resolved_method.strip().lower()
    resolved_scaler = scaler if scaler is not None else get_scaler_for_method(resolved_method)
    return ScalingSpec(method=resolved_method, scaler=resolved_scaler)
