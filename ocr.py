from threading import Thread

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


class GoogleOCR:

    def __init__(self):
        self.client = vision.ImageAnnotatorClient()

    def detect(self, frame):
        img_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        image = vision.Image(content=img_bytes)

        response = self.client.text_detection(image=image)

        if response.error.message:
            raise Exception(response.error.message)

        letters = []
        positions = []
        for text in response.text_annotations:
            left_top = vertex2tuple(text.bounding_poly.vertices[0])
            right_bottom = vertex2tuple(text.bounding_poly.vertices[2])
            text_area = abs(right_bottom[0] - left_top[0]) * abs(right_bottom[1] - left_top[1])
            position = np.array([(right_bottom[0] + left_top[0]) / 2, (right_bottom[1] + left_top[1]) / 2])
            text = text.description
            if len(text) != 1:
                continue
            if text_area < 500:
                continue

            letters.append(text)
            positions.append(position)
            frame = cv2.rectangle(frame, left_top, right_bottom, (0, 0, 255), 1)
            frame = cv2.putText(frame, text, left_top, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

        return frame, letters, positions
