from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class ModelRegistration:
    extension: str
    backend: str
    supports_tensorrt: bool = False
    requires_tensorrt: bool = False


@dataclass(frozen=True)
class ResolvedModelRegistration:
    extension: str
    backend: str
    model_name: str
    use_tensorrt: bool


class ModelRegistry:
    """Registry for model format -> backend resolution."""

    def __init__(self) -> None:
        self._registry: Dict[str, ModelRegistration] = {}

    @staticmethod
    def _normalize_extension(extension: str) -> str:
        ext = extension.lower().strip()
        if ext.startswith("."):
            ext = ext[1:]
        return ext

    def register(
        self,
        *,
        extension: str,
        backend: str,
        supports_tensorrt: bool = False,
        requires_tensorrt: bool = False,
    ) -> None:
        ext = self._normalize_extension(extension)
        self._registry[ext] = ModelRegistration(
            extension=ext,
            backend=backend,
            supports_tensorrt=supports_tensorrt,
            requires_tensorrt=requires_tensorrt,
        )

    def resolve(self, *, model_name: str, use_tensorrt: bool = False) -> ResolvedModelRegistration:
        extension = self._normalize_extension(Path(model_name).suffix)
        if not extension:
            raise ValueError(f"Model '{model_name}' does not include an extension")

        entry = self._registry.get(extension)
        if entry is None:
            allowed = ", ".join(sorted(self._registry))
            raise ValueError(
                f"Unsupported model extension '.{extension}'. Supported extensions: {allowed}"
            )

        if entry.requires_tensorrt and not use_tensorrt:
            raise TypeError(
                f"Model extension '.{extension}' requires TensorRT. "
                "Enable TensorRT mode to run this model."
            )

        if use_tensorrt and not entry.supports_tensorrt and not entry.requires_tensorrt:
            raise TypeError(
                f"TensorRT mode is not supported for '.{extension}' models"
            )

        resolved = entry
        if use_tensorrt and entry.supports_tensorrt:
            resolved = replace(entry, backend="trt")

        return ResolvedModelRegistration(
            extension=resolved.extension,
            backend=resolved.backend,
            model_name=model_name,
            use_tensorrt=use_tensorrt,
        )


def build_default_model_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register(extension="pt", backend="pt")
    registry.register(extension="onnx", backend="onnx", supports_tensorrt=True)
    registry.register(extension="engine", backend="trt", requires_tensorrt=True)
    return registry
