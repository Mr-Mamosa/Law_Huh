# backend/rag_api/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("ask", views.ask, name="ask"),
    path("status", views.status, name="status"),
]