from django.urls import path
from .views import CaptionAPIView

urlpatterns = [
    path('', CaptionAPIView.as_view(), name="CaptionAPIView"),
]
