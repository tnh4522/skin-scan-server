import base64
import os
import re
import time
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status


class UploadImageView(APIView):
    def post(self, request, *args, **kwargs):
        """
        Nhận dataUrl base64 từ client, decode và lưu vào media/uploads
        """
        data_url = request.data.get('dataUrl')
        if not data_url:
            return Response({"message": "No dataUrl provided."}, status=status.HTTP_400_BAD_REQUEST)

        # data_url dạng: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA..."
        pattern = r'^data:(image/\w+);base64,(.+)$'
        match = re.match(pattern, data_url)
        if not match:
            return Response({"message": "Invalid dataUrl format."}, status=status.HTTP_400_BAD_REQUEST)

        image_type = match.group(1)  # vd: "image/png"
        base64_str = match.group(2)

        # Tạo tên file (tùy ý), ở đây dùng timestamp
        extension = image_type.split('/')[-1]  # png/jpg/jpeg
        file_name = f"{int(time.time())}.{extension}"

        # Decode base64 -> bytes
        image_data = base64.b64decode(base64_str)

        # Đường dẫn lưu file: media/uploads/...
        save_path = os.path.join(settings.MEDIA_ROOT, 'uploads', file_name)

        try:
            # Tạo thư mục nếu chưa có
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Ghi file xuống disk
            with open(save_path, 'wb') as f:
                f.write(image_data)

            # Nếu muốn lưu thông tin vào Model UploadedImage thì:
            # from .unet import UploadedImage
            # uploaded_obj = UploadedImage.objects.create(image=f"uploads/{file_name}")
            # (Có thể trả về thông tin model)

            return Response({
                "status": 200,
                "message": "Image saved!",
                "file_name": file_name
            }, status=status.HTTP_201_CREATED)
        except Exception as e:
            print("Error saving image:", e)
            return Response({"message": "Internal server error."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
