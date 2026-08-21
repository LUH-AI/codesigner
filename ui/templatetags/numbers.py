"""`{{ x|sig }}` — a number as a reader should see it.

Django's own `floatformat` counts decimal places, which is the wrong unit for
this app's numbers: the same four places that suit a score between 0 and 1 write
a small hyperparameter effect as 0.0000. See `ui.formatting` for the rule.
"""

from django import template

from ..formatting import sigfigs

register = template.Library()


@register.filter(name="sig")
def sig(value, digits=None):
    """*value* to four significant figures, or *digits* if given."""
    return sigfigs(value) if digits is None else sigfigs(value, int(digits))
