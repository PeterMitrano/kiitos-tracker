import re

import cv2
import numpy as np
from google.cloud import vision

import annotation

MAX_BASELINE_ANGLE_DEG = 30
MAX_TEXT_AREA = 500

bbox_annotation_color = (0, 0, 255)
baseline_annotation_color = (0, 255, 0)


def vertex2tuple(v):
    return v.x, v.y


def fix_common_misdetections(text):
    common_misdetections = {
        '0': 'O',
        '2': 'Z',
    }
    return common_misdetections.get(text, text)


def in_bbox(p, bbox):
    return bbox[0, 0] < p[0] < bbox[1, 0] and bbox[0, 1] < p[1] < bbox[1, 1]


def filter_non_cards(text_and_vertices, workspace_bbox):
    letters = []
    positions = []
    valid_text_and_vertices = []
    for text, vertices in text_and_vertices:
        left_top = vertex2tuple(vertices[0])
        right_bottom = vertex2tuple(vertices[2])
        left_bottom = vertex2tuple(vertices[3])
        text_w = abs(right_bottom[0] - left_top[0])
        text_h = abs(right_bottom[1] - left_top[1])
        text_area = text_w * text_h
        position = np.array([(right_bottom[0] + left_top[0]) / 2, (right_bottom[1] + left_top[1]) / 2])

        # NOTE: assumes that the camera is aligned with the word in a certain way
        slope_in_img_frame = np.array(right_bottom) - np.array(left_bottom)
        angle_in_img_frame = np.rad2deg(np.arctan2(slope_in_img_frame[1], slope_in_img_frame[0]))
        if abs(angle_in_img_frame) > MAX_BASELINE_ANGLE_DEG:
            continue
        if len(text) != 1:
            print('!!!!!', text)
            continue
        if text_area < MAX_TEXT_AREA:
            continue
        if not in_bbox(position, workspace_bbox):
            continue
        if not re.search(r'([a-z]|[A-Z])', text):
            continue

        letter = fix_common_misdetections(text)

        letters.append(letter)
        positions.append(position)
        valid_text_and_vertices.append((text, vertices))

    return letters, positions, valid_text_and_vertices


def annotate(input_img, text_and_vertices):
    annotated = input_img.copy()
    for letter, vertices in text_and_vertices:
        left_top = vertex2tuple(vertices[0])
        right_bottom = vertex2tuple(vertices[2])
        left_bottom = vertex2tuple(vertices[3])
        text_pos = (left_top[0] + 5, left_top[1] - 25)

        annotated = cv2.rectangle(annotated, left_top, right_bottom, bbox_annotation_color, 1)
        annotated = cv2.putText(annotated, letter, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_annotation_color, 1)
        annotation.text(annotated, letter, text_pos, 0.5, bbox_annotation_color, 1)
        annotated = cv2.line(annotated, left_bottom, right_bottom, baseline_annotation_color, 5)
    return annotated


class GoogleOCR:

    def __init__(self):
        self.client = vision.ImageAnnotatorClient()

    def detect(self, input_img, workspace_bbox):
        adaptiveThresh = input_img.copy()

        img_bytes = cv2.imencode('.jpg', adaptiveThresh)[1].tobytes()
        image = vision.Image(content=img_bytes)

        response = self.client.document_text_detection(image=image)
        if response.error.message:
            raise Exception(response.error.message)

        document = response.full_text_annotation
        text_and_vertices = []
        for page in document.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        for symbol in word.symbols:
                            if symbol.confidence < 0.4:
                                continue
                            text_and_vertices.append((symbol.text, symbol.bounding_box.vertices))

        letters, positions, valid_text_and_vertices = filter_non_cards(text_and_vertices, workspace_bbox)
        annotated = annotate(input_img, text_and_vertices)

        return annotated, letters, positions
