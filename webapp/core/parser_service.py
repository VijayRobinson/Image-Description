
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


CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


class SimpleConfig(mrcnn.config.Config):
    NAME = "coco_inference"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = len(CLASS_NAMES)


BEHIND_THRES = -0.2
FRONT_THRES = 0.2
BEHIND_WORD = "behind"
FRONT_WORD = "front of"
NEAR_WORD = "near"


class ParserService():

    def __init__(self):
        self.model = model = mrcnn.model.MaskRCNN(
            mode="inference", config=SimpleConfig(), model_dir=os.getcwd())
        model.load_weights(OBJ_MODEL_PATH, by_name=True)

    def get_objects_and_locations(self, image_path):
        image = np.asarray(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rois = self.model.detect([image], verbose=0)[0]
        return rois

    def get_angle(self, node_a, node_b):
        x1, y1 = node_a.end
        x2, y2 = node_b.end
        tan = (y2-y1)/(x2-x1)
        return tan

    def distance(self, node_a, node_b):
        return np.linalg.norm(np.array(node_a.end) - np.array(node_b.end))

    def get_position_word(self, node_a, node_b):
        angle = self.get_angle(node_a, node_b)
        if angle < BEHIND_THRES:
            return self.prepare_sentence(node_b, node_a, BEHIND_WORD)
        elif angle > FRONT_THRES:
            return self.prepare_sentence(node_b, node_a, FRONT_WORD)
        return self.prepare_sentence(node_a, node_b, NEAR_WORD)

    def prepare_sentence(self, node_a, node_b, join_word):
        return f"{node_a.reference_name} is {join_word} {node_b.reference_name}"

    def process_sentence_objects(self, objects):
        res = []

        if len(objects) < 1:
            return ["Couldn't find any object..."]

        if len(objects) == 1:
            return [(f"a {objects[0].class_name} is in front of you.")]
        i = 0
        while i < len(objects)-1:
            node_a = objects[i]
            node_b = objects[i+1]
            res.append((self.get_position_word(node_a, node_b)))
            i += 1
        return res

    def add_node_reference_name(self, objects):
        objects_count_dict = {}
        for each in objects:
            if each.class_id in objects_count_dict:
                objects_count_dict[each.class_id] += 1
            else:
                objects_count_dict[each.class_id] = 1
            each.reference_name = each.class_name + " " + \
                str(objects_count_dict[each.class_id])

    def create_nodes(self, res):
        objects = []
        for i in range(len(res['rois'])):
            class_name = CLASS_NAMES[res['class_ids'][i]]
            class_id = res['class_ids'][i]
            points = res['rois'][i]
            start = (points[1], points[0])
            end = (points[3], points[2])
            score = res['scores'][i]
            ob = Node(start, end, class_id, class_name)
            if score > 0.9:
                objects.append(ob)
        self.add_node_reference_name(objects)
        return objects

    def process_objects(self, image_path):
        res = self.get_objects_and_locations(image_path)
        objects = self.create_nodes(res)
        return self.process_sentence_objects(objects)
