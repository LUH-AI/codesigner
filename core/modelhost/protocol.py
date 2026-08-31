"""The wire format between the application and a model in its own environment.

One JSON object per line, UTF-8, over the child's stdin and stdout. Requests
carry a monotonic ``id`` and every reply echoes it; exactly one request is in
flight at a time, so a mismatched id means the conversation has desynchronised
and is not something to recover from.

The child speaks first, unsolicited, with ``hello`` — which is what makes
"describe this model" a separate, short-lived invocation: start it, read the
greeting, stop. That is how the application learns a model's name and search
space without importing user code into a web request.

Two rules that are not obvious from the message list:

**Nothing the child sends is ever unpickled.** Predictions come back as a JSON
list. A binary array frame would need ``allow_pickle`` for exactly the
string-label case that matters, which would hand arbitrary code execution back
to the process this whole design exists to isolate. Probabilities travel the
same way — they are plain floats, so a numeric ``.npy`` would be safe here, but
one message shape is worth more than the bytes until a dataset is big enough to
notice.

**The validation labels are never sent for the fold being scored.** The child
receives the whole feature matrix once, the fold divisions, and the *training*
labels of each fold. For a single holdout — one fold — that is the strong
statement: the child never sees the answers at all. For k-fold it is weaker,
because every row trains in k-1 folds and the union of what is sent is every
label; see `core.splits` for why that trade was taken.
"""

#: Bumped when the message set changes incompatibly. The child reports the
#: version it speaks in `hello`, and a mismatch is refused rather than guessed at.
#:
#: Deliberately *not* bumped for optional keys. The check is exact equality, so a
#: bump is a cliff rather than a slope — every environment already built would
#: stop answering until it was rebuilt. A key that a reader may ignore and a
#: writer may omit costs nothing to add, and `capabilities` below is how the two
#: sides find out what each other can do without either having to guess from a
#: number.
PROTOCOL_VERSION = 2

#: What a child says it can do beyond the minimum, in `hello`. Absent or empty
#: means "labels only", which is every model written before probabilities
#: existed and every model that simply does not offer them.
CAP_PROBA = "proba"

# Child → parent
HELLO = "hello"     # unsolicited, first: name, config space, protocol version, capabilities
READY = "ready"     # the split has been loaded
RESULT = "result"   # predictions for one configuration, and probabilities if asked
ERROR = "error"     # something failed; `kind` says what sort

# Parent → child
INIT = "init"          # where the dataset is, how it is divided, and the training labels
TRIAL = "trial"        # a configuration to evaluate, on one fold; `want_proba` asks for probabilities
SHUTDOWN = "shutdown"  # leave cleanly

#: `error.kind` values. `load` and `protocol` end the run; `trial` fails one
#: configuration and the search continues.
KIND_LOAD = "load"
KIND_TRIAL = "trial"
KIND_PROTOCOL = "protocol"

#: Files written into the shared directory named by `init`. The feature matrix
#: goes once, whole, as .npy for size and with pickling disabled — so a dataset
#: with a non-numeric feature column is refused rather than silently pickled.
#: Each fold's row indices go the same way; they are integers and there can be
#: a great many of them. Only the labels travel inside the `init` message as
#: JSON, because they are commonly strings and pickling them is the thing being
#: avoided.
X_FILE = "X.npy"


def fold_file(index: int, part: str) -> str:
    """The .npy holding one fold's row indices. *part* is "train" or "val"."""
    return f"fold_{index}_{part}.npy"
