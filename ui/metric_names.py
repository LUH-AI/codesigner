"""`metric_title` shows a metric's title in the reader's language. `core.metrics`
imports no Django, so its titles are registered here for `makemessages`.
Abbreviations such as RMSE are the same in every language."""

from django.utils.translation import gettext as _
from django.utils.translation import gettext_noop

#: The `core.metrics` titles a translation changes.
TITLES = (gettext_noop("Accuracy"), gettext_noop("F1"), gettext_noop("Precision"),
          gettext_noop("Recall(macro)"), gettext_noop("Balanced accuracy"),
          gettext_noop("Log loss"))


def metric_title(metric) -> str:
    return _(metric.title)
