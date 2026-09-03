"""A home for permissions that are not about a row.

`use_custom_models` is a property of an account, not of any particular
experiment, so there is no model it naturally hangs off. Django's answer is an
unmanaged model with no default permissions: no table is created, but the
`Permission` rows are, so the permission appears in the admin's user editor and
`user.has_perm("access.use_custom_models")` works.
"""

from django.db import models


class AccessPermissions(models.Model):
    class Meta:
        managed = False
        default_permissions = ()
        permissions = [
            ("use_custom_models",
             "Can upload and run custom models (arbitrary code execution)"),
        ]
        verbose_name_plural = "Access permissions"
