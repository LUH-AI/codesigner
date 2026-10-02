"""The contract a Codesigner model implements.

This is a standalone package, separate from the application, because a model
file is imported in *its own* environment — where the application does not
exist and cannot be reached for. It has no dependencies of its own, so
installing it constrains nothing about what a model may use.

    from codesigner_model import BaseModel

``BaseModel`` is the tunable contract (:class:`OptimizableBaseModel`).
"""

from .base import BaseModel, ModelBase, OptimizableBaseModel, OutputBaseModel

__all__ = ["BaseModel", "ModelBase", "OptimizableBaseModel", "OutputBaseModel"]
__version__ = "0.1.0"
