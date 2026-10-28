"""`core.modelhost` runs a user's model as a subprocess under the model's own
interpreter, over a pipe, since the model may need libraries the app lacks. The
subprocess executes `harness.py`, where `core.modelhost` is unavailable.
"""

from .client import HARNESS, describe, model_session
