"""Shared per-trial timing for optimizers.

`timed_evaluation` wraps a single trial's model evaluation and produces the
nine SMAC-native runhistory fields, so every optimizer records the same shape
whether or not it has its own timing. An optimizer opts in by wrapping its
`model.train_evaluate(...)` call and passing the resulting dict to
`TrialCollector.record(run_info=...)`; it is not required to.
"""

import time
from contextlib import contextmanager

# The per-trial fields, matching SMAC's runhistory data entries. Single source
# of truth for both the builder here and the (de)serializer in base.py.
RUN_INFO_KEYS = (
    "instance", "seed", "budget", "time", "cpu_time",
    "status", "starttime", "endtime", "additional_info",
)

# SMAC's StatusType.SUCCESS as the stored int (base stays SMAC-import-free).
_STATUS_SUCCESS = 1


@contextmanager
def timed_evaluation(seed, budget=None, instance=None):
    """Measure one trial evaluation; yield a dict filled on exit.

    `time` is the monotonic wall-clock duration (perf_counter); `cpu_time` is
    this thread's CPU time (isolating it from other concurrent runs, though it
    undercounts BLAS worker threads); `starttime`/`endtime` are POSIX
    timestamps. `instance`/`budget` are None for our single-instance,
    non-multi-fidelity runs.
    """
    run_info = {}
    starttime = time.time()
    perf0 = time.perf_counter()
    cpu0 = time.thread_time()
    try:
        yield run_info
    finally:
        run_info.update({
            "instance": instance,
            "seed": seed,
            "budget": budget,
            "time": time.perf_counter() - perf0,
            "cpu_time": time.thread_time() - cpu0,
            "status": _STATUS_SUCCESS,
            "starttime": starttime,
            "endtime": time.time(),
            "additional_info": {},
        })
