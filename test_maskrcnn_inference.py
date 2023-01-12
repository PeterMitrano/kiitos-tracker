import pathlib
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def viz(viz_img, viz_inputs, seperate_masks=False, n_show=5):
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
        letter = chr(label - 1 + ord('A'))
        main_ax.text(x=int(box[0]), y=int(box[1]), s=f"{letter} {score:.1f}")


def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


def load_model():
    num_classes = 27
    model_path = pathlib.Path("model-4.pt")
    model = get_instance_segmentation_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def filter_nms(real_prediction):
    keep_indices = torchvision.ops.nms(real_prediction['boxes'], real_prediction['scores'], 0.2)
    keys = ['boxes', 'scores', 'masks', 'labels']
    for k in keys:
        real_prediction[k] = real_prediction[k][keep_indices]


def main():
    model = load_model()
    outdir = pathlib.Path('detections')
    outdir.mkdir(exist_ok=True)

    real_test_dir = pathlib.Path("test_images2/")
    for real_test_image_path in real_test_dir.iterdir():
        real_test_img = Image.open(real_test_image_path)

        resize_t = torchvision.transforms.Resize([480, 640])

        t0 = perf_counter()
        real_test_img_np = np.transpose(np.array(real_test_img), [2, 0, 1])
        real_img_tensor = torch.tensor(real_test_img_np).float()
        real_img_tensor = real_img_tensor / real_img_tensor.max()
        real_img_tensor_resized = resize_t(real_img_tensor)

        with torch.no_grad():
            real_prediction = model([real_img_tensor_resized])
        real_prediction = real_prediction[0]
        dt = perf_counter() - t0
        print(f'{dt:.2f}s')

        filter_nms(real_prediction)

        viz(real_img_tensor_resized, real_prediction, n_show=10)
        outpath = outdir / f'{real_test_image_path.stem}-detection.png'
        plt.savefig(outpath.as_posix())
        plt.show()


if __name__ == '__main__':
    main()
