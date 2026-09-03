"""The model contract, re-exported from the SDK.

The contract lives in the standalone, dependency-free ``codesigner_model``
package (``model_sdk/`` in this repo) because a model file is imported in *its
own* environment, where ``core`` does not exist and cannot be reached for.
Defining the class twice would give two unrelated ABCs and ``issubclass`` would
be False across the seam — so there is exactly one class, and this module only
preserves the import path the application has always used.

New model files should ``from codesigner_model import BaseModel``.
"""

from codesigner_model import BaseModel

__all__ = ["BaseModel"]
