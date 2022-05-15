
from core.parser_service import CLASS_NAMES, SimpleConfig
from . import mrcnn
from .mrcnn import config
from .mrcnn import model
from .mrcnn import visualize
import cv2
import os
import numpy as np

from webapp.settings import OBJ_MODEL_PATH


class Node:
    def __init__(self, start, end, class_id, class_name):
        self.start = start
        self.end = end
        self.class_id = class_id
        self.class_name = class_name

    def get_area(self):
        height = self.end[1] - self.start[1]
        width = self.end[0] - self.start[0]
        return width * height

    def to_json(self):
        mdict = {
            "start_x": self.start[0],
            "start_y": self.start[1],
            "end_x": self.end[0],
            "end_y": self.end[1],
            "class_name": self.class_name,
            "class_id": self.class_id
        }
        return mdict


class DirectionService():

    def __init__(self):
        self.model = model = mrcnn.model.MaskRCNN(
            mode="inference", config=SimpleConfig(), model_dir=os.getcwd())
        model.load_weights(OBJ_MODEL_PATH, by_name=True)

    def get_objects_and_locations(self, image_path):
        image = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
        rois = self.model.detect([image], verbose=0)[0]
        return rois

    def get_pos_node(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_width, im_height = image.shape[1], image.shape[0]
        cur_pos = (im_height, int(im_width/2))
        pos_node = Node(cur_pos, cur_pos, -1, "current_position")
        return pos_node

    def get_angle(self, node_a, node_b):
        x1, y1 = node_a.end
        x2, y2 = node_b.end
        tan = (y2-y1)/(x2-x1)
        return tan

    def distance(self, node_a, node_b):
        return np.linalg.norm(np.array(node_a.end) - np.array(node_b.end))

    def get_nodes(self, res):
        objects = []
        for i in range(len(res['rois'])):
            class_name = CLASS_NAMES[res['class_ids'][i]]
            class_id = res['class_ids'][i]
            points = res['rois'][i]
            start = (points[1], points[0])
            end = (points[3], points[2])
            score = res['scores'][i]
            ob = Node(start, end, class_id, class_name)
            if score > 0.8:
                objects.append(ob)
        return objects

    def parse_direction(self, pos_node, old_target_node, target_node):
        angle = self.get_angle(pos_node, target_node)
        movement = ""
        if angle < -1.0:
            movement = "right"
        elif angle > 1.0:
            movement = "left"
        else:
            movement = "front"
        dis_str = ""
        if old_target_node is not None:
            if old_target_node.get_area() < target_node.get_area():
                dis_str = "getting closer to"
            else:
                dis_str = "moving away from"
        if dis_str == "":
            return f"move {movement}"
        return f" Your are {dis_str} {CLASS_NAMES[target_node.class_id]},move {movement} "

    def process(self, image, target_node, target_class):
        image = np.asarray(image)
        pos_node = self.get_pos_node(image)
        res = self.get_objects_and_locations(image)
        objects = self.get_nodes(res)
        for each in objects:
            if each.class_id == CLASS_NAMES.index(target_class):
                return self.parse_direction(pos_node, target_node, each),  each
        return "Unable to find the object...", pos_node
