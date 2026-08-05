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
to the process this whole design exists to isolate.

**The validation labels are never sent.** The child receives X_train, y_train
and X_val. It cannot see the answers, so it cannot score itself, and every
model is measured by the application's own metrics.
"""

#: Bumped when the message set changes incompatibly. The child reports the
#: version it speaks in `hello`, and a mismatch is refused rather than guessed at.
PROTOCOL_VERSION = 1

# Child → parent
HELLO = "hello"     # unsolicited, first: name, config space, protocol version
READY = "ready"     # the split has been loaded
RESULT = "result"   # predictions for one configuration
ERROR = "error"     # something failed; `kind` says what sort

# Parent → child
INIT = "init"          # where the split is, and how many validation rows to expect
TRIAL = "trial"        # a configuration to evaluate
SHUTDOWN = "shutdown"  # leave cleanly

#: `error.kind` values. `load` and `protocol` end the run; `trial` fails one
#: configuration and the search continues.
KIND_LOAD = "load"
KIND_TRIAL = "trial"
KIND_PROTOCOL = "protocol"

#: Filenames written into the shared directory named by `init`. The feature
#: arrays go as .npy for size, with pickling disabled — so a dataset with a
#: non-numeric feature column is refused rather than silently pickled. The
#: labels travel inside the `init` message as JSON, because they are commonly
#: strings and pickling them is exactly what we are avoiding.
X_TRAIN_FILE = "X_train.npy"
X_VAL_FILE = "X_val.npy"
