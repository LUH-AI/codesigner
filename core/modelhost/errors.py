

class ModelHostError(Exception):
    """`detail` contains the model's own traceback, when the model sent one.
    """

    def __init__(self, *args, detail: str = ""):
        super().__init__(*args)
        self.detail = detail


class ModelProcessError(ModelHostError):
    """`ModelProcessError` ends the run: the model process failed to start,
    died, or broke the protocol.
    """


class ModelTrialError(ModelHostError):
    """The model raised on one configuration; the trial fails and the run
    continues.
    """


class TrialTimeout(ModelHostError):
    """The model exceeded one trial's time limit."""


class TrialCancelled(ModelHostError):
    """The run was cancelled mid-trial; the loop stops and keeps the finished
    trials.
    """
