import pathlib

import numpy as np
from PIL import Image

from kiitos.upload_for_labeling import upload_to_labelbox, make_labelbox_client


def main():
    img_np = np.random.uniform(0, 255, size=[320, 480, 3]).astype(np.uint8)
    image_path = pathlib.Path("test.png")
    Image.fromarray(img_np).save(image_path)

    labelbox_client = make_labelbox_client()
    upload_to_labelbox(labelbox_client, image_path)


if __name__ == '__main__':
    main()
