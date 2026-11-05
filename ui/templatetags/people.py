"""`shown` names a person the way other people see them (`.shown_name`), and
`shown_with_email` adds their email address, for the group page, where people
are managed and two may share a display name."""

from django import template

from access.models import shown_name

register = template.Library()


@register.filter
def shown(user) -> str:
    return shown_name(user)


@register.filter
def shown_with_email(user) -> str:
    name = shown_name(user)
    email = getattr(user, "email", "")
    return f"{name} ({email})" if email and name != email else name
