"""`protocol` defines newline-delimited JSON over the child's stdin and stdout.
Each request carries an ``id`` that the reply echoes; with one request in
flight, a mismatched id ends the conversation. The child speaks first with
``hello``, so the application learns a model's name and space without importing
user code. The parent never unpickles child output, and the scored fold's
validation labels never reach the child, though k-fold sends every label across
folds (see `core.splits`).
"""

#: A mismatched version is refused. Optional keys never bump `PROTOCOL_VERSION`,
#: since the exact-equality check would break every built environment;
#: `capabilities` announces optional features instead.
PROTOCOL_VERSION = 2

#: A missing or empty list means labels only.
CAP_PROBA = "proba"

HELLO = "hello"     # first, unsolicited; carries capabilities
READY = "ready"     # split loaded
RESULT = "result"   # predictions, plus probabilities if asked
ERROR = "error"     # `kind` says what failed

INIT = "init"          # dataset location, folds, training labels
TRIAL = "trial"        # one fold; `want_proba` asks for probabilities
SHUTDOWN = "shutdown"

#: `load` and `protocol` end the run; `trial` fails one configuration.
KIND_LOAD = "load"
KIND_TRIAL = "trial"
KIND_PROTOCOL = "protocol"

#: `X.npy` stores the features without pickling, and a non-numeric feature
#: column is refused. Fold indices also go as .npy; only labels travel as JSON
#: in `init`, since labels are often strings.
X_FILE = "X.npy"


def fold_file(index: int, part: str) -> str:
    """*part* is "train" or "val"."""
    return f"fold_{index}_{part}.npy"
