"""`evaluate_trial` wraps every trial in `timed_evaluation`, so all optimizers
record SMAC's nine runhistory fields.
"""

import time
from contextlib import contextmanager

RUN_INFO_KEYS = (
    "instance", "seed", "budget", "time", "cpu_time",
    "status", "starttime", "endtime", "additional_info",
)

# The ints copy SMAC's `StatusType`, keeping base.py free of SMAC imports.
STATUS_SUCCESS = 1
STATUS_CRASHED = 2
STATUS_TIMEOUT = 3

#: `CANCELLED` marks an abandoned attempt in `run_info`, and
#: `TrialCollector.record` drops the attempt. `CANCELLED` stays outside the
#: `STATUS_` ints because `smac_optimizer` passes those to `StatusType`, which
#: would raise on a new value.
CANCELLED = "cancelled"

#: `evaluate_trial` returns the last fold's model under `FITTED` when asked to
#: keep it, and `TrialCollector.record` removes it before recording.
FITTED = "_fitted"



@contextmanager
def timed_evaluation(seed, budget=None, instance=None):
    """`time` is perf_counter wall time, and `cpu_time` is the calling thread's
    CPU time, which undercounts BLAS workers.
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
            "status": STATUS_SUCCESS,
            "starttime": starttime,
            "endtime": time.time(),
            "additional_info": {},
        })
