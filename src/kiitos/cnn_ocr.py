import numpy as np
import torch

from kiitos.annotate import annotate
from utils.general import non_max_suppression

MAX_BASELINE_ANGLE_DEG = 30
MAX_TEXT_AREA = 500


def label_to_letter(label):
    return chr(label - 1 + ord('A'))


def box_to_vertices(box):
    x1, y1, x2, y2 = box
    left_top = (x1, y1)
    right_top = (x2, y1)
    right_bottom = (x2, y2)
    left_bottom = (x1, y2)
    vertices = [left_top, right_top, right_bottom, left_bottom]
    return vertices


def get_predictions_yolo(model, input_img):
    test_img_np = np.transpose(input_img, [2, 0, 1])
    img_tensor = torch.tensor(test_img_np).float()
    img_tensor /= 255.0
    img_tensor = img_tensor.unsqueeze(0)
    with torch.no_grad():
        prediction = model(img_tensor)
    predictions = prediction[0]
    predictions = non_max_suppression(predictions, conf_thres=0.7)
    prediction = predictions[0]
    boxes = prediction[:, :4]
    scores = prediction[:, 4].numpy()
    labels = prediction[:, 5].numpy().astype(int) + 1
    return {
        'boxes': boxes,
        'scores': scores,
        'labels': labels,
    }


def predictions_to_text_and_vertices(real_prediction):
    text_and_vertices = []
    for box, label in zip(real_prediction['boxes'], real_prediction['labels']):
        letter = label_to_letter(label)
        vertices = box_to_vertices(box.numpy().squeeze().astype(int))
        text_and_vertices.append((letter, vertices))
    return text_and_vertices


def load_yolov7():
    model = torch.load('best_real2.pt', map_location='cpu')
    model = model['model'].to(dtype=torch.float32)
    model.eval()
    return model


class CNNOCR:

    def __init__(self):
        self.model = load_yolov7()

    def detect(self, input_img, workspace_bbox):
        predictions = get_predictions_yolo(self.model, input_img)

        text_and_vertices = predictions_to_text_and_vertices(predictions)

        letters, positions, valid_text_and_vertices = filter_detections(text_and_vertices, workspace_bbox)
        annotated = annotate(input_img, valid_text_and_vertices)

        return annotated, letters, positions


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
