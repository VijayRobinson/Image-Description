from django.shortcuts import render
from numpy import imag

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny

from core.caption_service import CaptionService
from PIL import Image

from core.parser_service import ParserService


class CaptionAPIView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        image = request.FILES.get("image")
        caption_service = CaptionService()
        parser_service = ParserService()
        context = {
            "caption": caption_service.get_caption(Image.open(image)),
            "description": parser_service.process_objects(Image.open(image))
        }
        return Response(data=context)


class DirectionAPIView(APIView):

    permission_classes = [AllowAny]

    def post(self, request):
        pass
