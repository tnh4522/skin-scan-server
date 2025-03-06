from django.contrib import admin
from django.urls import path
from upload.views import UploadImageView
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/upload/', UploadImageView.as_view(), name='upload_image'),
]

# Chỉ dùng cho môi trường dev để serve file media
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
