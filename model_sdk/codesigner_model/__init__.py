"""`codesigner_model` holds the model contract as a standalone package without
dependencies, because a model file imports it in its own environment.
"""

from .base import BaseModel, ModelBase, OptimizableBaseModel, OutputBaseModel
__version__ = "0.1.0"
