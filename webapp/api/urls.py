from django.urls import path
from .views import CaptionAPIView, DirectionAPIView

urlpatterns = [
    path('caption', CaptionAPIView.as_view(), name="CaptionAPIView"),
    path('direction', DirectionAPIView.as_view(), name="DirectionAPIView")
]
