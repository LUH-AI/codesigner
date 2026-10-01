"""Data Handling's pages, under the experiment they handle the data of.

The full path is spelled out here, `<int:pk>` included, rather than mounted
under a prefix: the URLconf audit in tests/ui/test_permissions.py finds
experiment routes by that segment, and a route of this app should have to pass
it like any other.
"""

from django.urls import path

from . import views

app_name = "datahandling"

urlpatterns = [
    path("experiments/<int:pk>/data/", views.data_overview, name="data_overview"),
    path("experiments/<int:pk>/data/<slug:section>/", views.data_section, name="data_section"),
]
