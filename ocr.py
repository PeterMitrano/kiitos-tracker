import boto3
import cv2
import numpy as np
from google.cloud import vision


class AWS_OCR:
    def __init__(self):
        self.textract_client = boto3.client('textract')

    def detect(self, frame):
        img_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        res = self.textract_client.detect_document_text(Document={'Bytes': img_bytes})
        blocks = res['Blocks']

        frame_h, frame_w = frame.shape[0:2]
        print(len(blocks))
        for block in blocks:
            if block['BlockType'] == 'WORD':
                bbox = block['Geometry']['BoundingBox']
                text_w = bbox['Width']
                text_h = bbox['Height']
                text = block['Text']
                left_top = (int(bbox['Left'] * frame_w), int(bbox['Top'] * frame_h))
                right_bottom = (int((bbox['Left'] + text_w) * frame_w), int((bbox['Top'] + text_h) * frame_h))
                frame = cv2.rectangle(frame, left_top, right_bottom, (0, 0, 255), 1)
                frame = cv2.putText(frame, text, left_top, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

        return frame


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
            continue
        if text_area < 500:
            continue

        # NOTE: assumes that the camera is aligned with the word in a certain way
        slope_in_img_frame = np.array(right_bottom) - np.array(left_bottom)
        angle_in_img_frame = np.rad2deg(np.arctan2(slope_in_img_frame[1], slope_in_img_frame[0]))
        if abs(angle_in_img_frame) > 10:
            continue

        letter = fix_common_misdetections(text)

        letters.append(letter)
        positions.append(position)
        annotated = cv2.rectangle(annotated, left_top, right_bottom, (0, 0, 255), 1)
        annotated = cv2.line(annotated, left_bottom, right_bottom, (0, 255, 0), 5)
        annotated = cv2.putText(annotated, letter, left_top, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
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

        response = self.client.text_detection(image=image)
        if response.error.message:
            raise Exception(response.error.message)
        text_and_vertices = [(a.description, a.bounding_poly.vertices) for a in response.text_annotations]

        annotated, letters, positions = filter_and_annotate(input_img, text_and_vertices)

        return annotated, letters, positions
