import cv2
import numpy as np
from google.cloud import vision

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


def filter_and_annotate(input_img, text_and_vertices):
    annotated = input_img.copy()
    letters = []
    positions = []
    for text, vertices in text_and_vertices:
        left_top = vertex2tuple(vertices[0])
        right_bottom = vertex2tuple(vertices[2])
        left_bottom = vertex2tuple(vertices[3])
        text_area = abs(right_bottom[0] - left_top[0]) * abs(right_bottom[1] - left_top[1])
        position = np.array([(right_bottom[0] + left_top[0]) / 2, (right_bottom[1] + left_top[1]) / 2])
        if len(text) != 1:
            print(text)
            continue
        if text_area < MAX_TEXT_AREA:
            continue

        # NOTE: assumes that the camera is aligned with the word in a certain way
        slope_in_img_frame = np.array(right_bottom) - np.array(left_bottom)
        angle_in_img_frame = np.rad2deg(np.arctan2(slope_in_img_frame[1], slope_in_img_frame[0]))
        if abs(angle_in_img_frame) > MAX_BASELINE_ANGLE_DEG:
            continue

        letter = fix_common_misdetections(text)

        letters.append(letter)
        positions.append(position)
        annotated = cv2.rectangle(annotated, left_top, right_bottom, bbox_annotation_color, 1)
        annotated = cv2.putText(annotated, letter, left_top, cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_annotation_color, 1)
        annotated = cv2.line(annotated, left_bottom, right_bottom, baseline_annotation_color, 5)
    return annotated, letters, positions


class GoogleOCR:

    def __init__(self):
        self.client = vision.ImageAnnotatorClient()

    def detect(self, input_img):
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
                            text_and_vertices.append((symbol.text, symbol.bounding_box.vertices))

        annotated, letters, positions = filter_and_annotate(input_img, text_and_vertices)
        return annotated, letters, positions
