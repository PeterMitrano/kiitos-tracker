import pathlib
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image

from cnn_ocr import load_model, filter_nms, viz_plt


def main():
    model_path = pathlib.Path("model-6.pt")
    model = load_model(model_path)

    outdir = pathlib.Path('detections')
    outdir.mkdir(exist_ok=True)

    real_test_dir = pathlib.Path("test_images3/")
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

        viz_plt(real_img_tensor_resized, real_prediction, n_show=10)
        outpath = outdir / f'{real_test_image_path.stem}-detection.png'
        plt.savefig(outpath.as_posix())
        plt.show()


if __name__ == '__main__':
    main()
