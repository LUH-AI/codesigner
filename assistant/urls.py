from django.urls import path

from . import views

app_name = "assistant"

urlpatterns = [
    path("assistant/message/", views.message, name="message"),
    path("assistant/dataset/", views.dataset, name="dataset"),
    path("assistant/approve/", views.approve, name="approve"),
    path("assistant/reset/", views.reset, name="reset"),
    path("assistant/window/", views.window, name="window"),
    path("assistant/page/", views.page, name="page"),
    path("assistant/acts/", views.acts, name="acts"),
    path("assistant/acted/", views.acted, name="acted"),
    path("assistant/conversations/<int:pk>/events/", views.events, name="events"),
]
