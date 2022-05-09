from django.shortcuts import render
from numpy import imag

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny

from core.caption_service import CaptionService
from PIL import Image

from core.parser_service import ParserService

from core.directiion_service import DirectionService, Node


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

        image = request.FILES.get("image")
        target_node = None
        is_first = request.data.get("is_first")
        target_class = request.data.get("target_class")
        if not int(is_first):
            start = (int(request.data.get("start_x")),
                     int(request.data.get("start_y")))
            end = (int(request.data.get("end_x")),
                   int(request.data.get("end_y")))
            class_name = request.data.get("class_name")
            class_id = int(request.data.get("class_id"))
            target_node = Node(start=start, end=end,
                               class_id=class_id, class_name=class_name)
        direction_service = DirectionService()
        direction, target = direction_service.process(Image.open(
            image), target_node=target_node, target_class=target_class)

        context = {
            "direction": direction,
            "target": target.to_json()
        }
        return Response(data=context)
