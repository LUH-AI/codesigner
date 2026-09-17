"""The models, optimizers, and metrics the application offers.

These are the choices presented in forms and used to reconstruct experiments
from a snapshot. Keys are the human-readable names stored in .ihpo files.

Defined in `core.registry` and re-exported here. The application's vocabulary is
unchanged; what moved is where the definitions live, so that a run executing
outside this process — a cluster job with no Django — resolves the same names
from the same place rather than from a copy that can drift.
"""

from core.registry import METRICS, MODELS, OPTIMIZERS

__all__ = ["MODELS", "OPTIMIZERS", "METRICS"]
