import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from kiitos.annotate import annotate
from utils.general import non_max_suppression


def label_to_letter(label):
    return chr(label - 1 + ord('A'))


def letter_to_label(letter):
    return ord(letter) - ord('A') + 1


def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


def load_model(model_path):
    num_classes = 27
    model = get_instance_segmentation_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def filter_nms(real_prediction):
    keep_indices = torchvision.ops.nms(real_prediction['boxes'], real_prediction['scores'], 0.25)
    keys = ['boxes', 'scores', 'masks', 'labels']
    for k in keys:
        real_prediction[k] = real_prediction[k][keep_indices]


def viz_plt(viz_img, viz_inputs, seperate_masks=False, n_show=5):
    n_label = len(viz_inputs['labels'])
    if n_show is None:
        n_show = n_label
    n_show = min(n_show, n_label)

    if seperate_masks and n_show > 0:
        fig, ax = plt.subplots(1, n_show + 1, figsize=(24, 3))
        main_ax = ax[0]
        mask_axes = ax[1:]
    else:
        plt.figure()
        main_ax = plt.gca()
        mask_axes = [main_ax] * n_show

    main_ax.imshow(viz_img.numpy().transpose(1, 2, 0))

    for i in range(n_show):
        box = viz_inputs['boxes'][i].cpu().numpy()
        mask = viz_inputs['masks'][i].cpu()
        mask = mask.squeeze().numpy()
        if seperate_masks:
            mask_axes[i].imshow(mask)
        score = 1
        if 'scores' in viz_inputs:
            score = float(viz_inputs['scores'][i].cpu().numpy())
        mask_axes[i].plot([box[0], box[2], box[2], box[0], box[0]], [box[1], box[1], box[3], box[3], box[1]],
                          linewidth=5, alpha=score)
        main_ax.plot([box[0], box[2], box[2], box[0], box[0]], [box[1], box[1], box[3], box[3], box[1]], linewidth=5,
                     alpha=score)
        label = int(viz_inputs['labels'][i])
        letter = chr(label - 1 + ord('a')).upper()
        main_ax.add_text(x=int(box[0]), y=int(box[1]), s=f"{letter} {score:.1f}", size='large')


def box_to_vertices(box):
    x1, y1, x2, y2 = box
    left_top = (x1, y1)
    right_top = (x2, y1)
    right_bottom = (x2, y2)
    left_bottom = (x1, y2)
    vertices = [left_top, right_top, right_bottom, left_bottom]
    return vertices


def get_predictions_maskrcnn(model, input_img):
    real_test_img_np = np.transpose(input_img, [2, 0, 1])
    real_img_tensor = torch.tensor(real_test_img_np).float()
    real_img_tensor = real_img_tensor / real_img_tensor.max()
    with torch.no_grad():
        real_prediction = model([real_img_tensor])
    real_prediction = real_prediction[0]
    filter_nms(real_prediction)
    return real_prediction


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
    for box, score, label in zip(real_prediction['boxes'], real_prediction['scores'], real_prediction['labels']):
        letter = label_to_letter(label)
        vertices = box_to_vertices(box.numpy().squeeze().astype(int))
        text_and_vertices.append((letter, vertices))
    return text_and_vertices


def load_yolov7():
    import torch
    model = torch.load('best.pt', map_location='cpu')
    model = model['model'].to(dtype=torch.float32)
    model.eval()
    return model


def load_maskrcnn():
    model_path = pathlib.Path("../model-6.pt")
    model = load_model(model_path)
    return model


class CNNOCR:

    def __init__(self):
        # model_path = pathlib.Path("model-6.pt")
        # self.model = load_model(model_path)
        self.model = load_yolov7()

    def detect(self, input_img, workspace_bbox):
        # predictions = get_predictions_maskrcnn(self.model, input_img)
        predictions = get_predictions_yolo(self.model, input_img)

        text_and_vertices = predictions_to_text_and_vertices(predictions)

        letters, positions, valid_text_and_vertices = filter_detections(text_and_vertices, workspace_bbox)
        annotated = annotate(input_img, valid_text_and_vertices)

        return annotated, letters, positions


MAX_BASELINE_ANGLE_DEG = 30
MAX_TEXT_AREA = 500


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
