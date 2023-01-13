import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


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

    # print('labels', viz_inputs['labels'])
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
        letter = chr(label - 1 + ord('a'))
        main_ax.text(x=int(box[0]), y=int(box[1]), s=f"{letter} {score:.1f}")


def box_to_vertices(box):
    x1, y1, x2, y2 = box
    left_top = (x1, y1)
    right_top = (x2, y1)
    right_bottom = (x2, y2)
    left_bottom = (x1, y2)
    vertices = [left_top, right_top, right_bottom, left_bottom]
    return vertices
