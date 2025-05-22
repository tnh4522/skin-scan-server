from django.urls import path
from .views import WrinkleDetectionView

urlpatterns = [
    path('detect/', WrinkleDetectionView.as_view(), name='detect_wrinkle'),
]