import cv2
import numpy as np

import annotation

MAX_BASELINE_ANGLE_DEG = 30
MAX_TEXT_AREA = 500

bbox_annotation_color = (0, 0, 255)
baseline_annotation_color = (0, 255, 0)


def in_bbox(vertices, bbox):
    for p in vertices:
        if not (bbox.x0 < p[0] < bbox.x1 and bbox.y0 < p[1] < bbox.y1):
            return False
    return True


def filter_detections(text_and_vertices, workspace_bbox):
    letters = []
    positions = []
    valid_text_and_vertices = []
    for letter, vertices in text_and_vertices:
        left_top = vertices[0]
        right_bottom = vertices[2]
        left_bottom = vertices[3]
        text_w = abs(right_bottom[0] - left_top[0])
        text_h = abs(right_bottom[1] - left_top[1])
        text_area = text_w * text_h
        position = np.array([(right_bottom[0] + left_top[0]) / 2, (right_bottom[1] + left_top[1]) / 2])

        # NOTE: assumes that the camera is aligned with the word in a certain way
        slope_in_img_frame = np.array(right_bottom) - np.array(left_bottom)
        angle_in_img_frame = np.rad2deg(np.arctan2(slope_in_img_frame[1], slope_in_img_frame[0]))
        if abs(angle_in_img_frame) > MAX_BASELINE_ANGLE_DEG:
            continue
        if text_area < MAX_TEXT_AREA:
            continue
        if not in_bbox(vertices, workspace_bbox):
            continue

        letters.append(letter)
        positions.append(position)
        valid_text_and_vertices.append((letter, vertices))

    return letters, positions, valid_text_and_vertices


def annotate(input_img, text_and_vertices):
    annotated = input_img.copy()
    for letter, vertices in text_and_vertices:
        left_top = vertices[0]
        right_bottom = vertices[2]
        left_bottom = vertices[3]
        text_pos = (left_top[0] + 5, left_top[1] - 15)

        annotated = cv2.rectangle(annotated, left_top, right_bottom, bbox_annotation_color, 1)
        annotation.text(annotated, letter, text_pos, 1, bbox_annotation_color, 3)
        annotated = cv2.line(annotated, left_bottom, right_bottom, baseline_annotation_color, 5)
    return annotated
