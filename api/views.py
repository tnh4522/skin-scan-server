import os
import torch
import numpy as np
from PIL import Image
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from .unet.swin_unetr import SwinUNETR  # or UNet if using that model
from .utils import load_checkpoint, preprocess_image


class WrinkleDetectionView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SwinUNETR(in_channels=4, out_channels=2)
        ckpt_path = os.path.join("model_weights", "stage2_swinunetr.pth")
        self.model = load_checkpoint(self.model, ckpt_path, self.device)
        self.model.eval()
        self.model.to(self.device)

    def post(self, request, *args, **kwargs):
        try:
            image_file = request.data['image']
            texture_file = request.data['texture']

            image = Image.open(image_file).convert("RGB")
            texture = Image.open(texture_file).convert("L")
            input_tensor = preprocess_image(image, texture).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor)
                mask = output.argmax(dim=1)[0].cpu().numpy()
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))

            response = Response(status=status.HTTP_200_OK)
            response["Content-Type"] = "image/png"
            mask_img.save(response, format="PNG")
            return response

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
