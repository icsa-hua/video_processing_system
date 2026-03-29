from __future__ import annotations

from obs_system.detection_module.dummy_predictor.stream_unified import UnifiedModelStreamer
from obs_system.detection_module.interface.model_registry import (
    ModelRegistry,
    ResolvedModelRegistration,
    build_default_model_registry,
)

from pathlib import Path
from typing import Any
from ultralytics.utils import DEFAULT_CFG


class StreamerFactory:
    """Builds streamer instances using model registry resolution."""

    def __init__(
        self,
        *,
        registry: ModelRegistry | None = None,
        cfg: Any = DEFAULT_CFG,
        overrides: dict | None = None,
        callbacks: Any = None,
    ) -> None:
        self._registry = registry or build_default_model_registry()
        self._cfg = cfg
        self._overrides = overrides or {}
        self._callbacks = callbacks

    def create(
        self,
        *,
        model_name: str,
        path_to_load: str | Path,
        use_tensorrt: bool,
        opt: str = "tracking",
    ) -> tuple[UnifiedModelStreamer, ResolvedModelRegistration]:
        resolved = self._registry.resolve(model_name=model_name, use_tensorrt=use_tensorrt)

        streamer = UnifiedModelStreamer(self._cfg, self._overrides, self._callbacks)
        streamer.setup_model(
            model_name=model_name,
            path_to_load=path_to_load,
            opt=opt,
            backend=resolved.backend,
        )
        return streamer, resolved
