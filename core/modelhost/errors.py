"""What can go wrong when a model lives in another process."""


class ModelHostError(Exception):
    """Base for every failure involving a model in its own process.

    `detail` carries the model's own traceback when the failure happened over
    there and it sent one. A traceback raised on this side would show the client
    waiting for a reply, which says nothing about why the model failed, so the
    remote one is the only one worth keeping.
    """

    def __init__(self, *args, detail: str = ""):
        super().__init__(*args)
        self.detail = detail


class ModelProcessError(ModelHostError):
    """The process or the conversation with it failed.

    The model could not be started, died, spoke nonsense, or answered the wrong
    question. This is about the host, not about the model's opinion of a
    configuration — it ends the run rather than the trial.
    """


class ModelTrialError(ModelHostError):
    """The model raised while evaluating one configuration.

    A fact about that configuration, not about the model: the run continues and
    the trial is recorded as failed.
    """


class TrialTimeout(ModelHostError):
    """The model stopped answering within the time allowed for one trial."""


class TrialCancelled(ModelHostError):
    """The run was cancelled while a trial was in flight.

    Distinct from a timeout because the loop should stop and hand back the
    trials it already has, rather than record this one as a failure.
    """
