"""`{{ x|sig }}` formats numbers by significant figures, because `floatformat`
writes small effects as 0.0000.
"""

from django import template

from ..formatting import sigfigs

register = template.Library()


@register.filter(name="sig")
def sig(value, digits=None):
    """`sig` rounds to *digits*, default four."""
    return sigfigs(value) if digits is None else sigfigs(value, int(digits))
