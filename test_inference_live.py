import pathlib
from time import perf_counter

import PIL.Image
import cv2
import numpy as np
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

from cnn_ocr import get_predictions_yolo, load_yolov7, predictions_to_text_and_vertices
from ocr import annotate
from video_capture import CaptureManager


def visualize_predictions_from_dataset(img, predictions):
    doors_metadata = MetadataCatalog.get('kiitos_train')
    v = Visualizer(test_img, doors_metadata, scale=0.5)
    out = v.draw_instance_predictions(predictions['instances'].to("cpu"))
    annotated_img = out.get_image()
    cv2.imshow(annotated_img)


def main():
    # model = load_maskrcnn()
    # model = load_detectron2()
    model = load_yolov7()

    save_dir = pathlib.Path("saved_from_live")
    save_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(0)
    manager = CaptureManager(cap)
    while True:
        if manager.last_frame is None:
            continue

        np_img = manager.last_frame.copy()

        pil_img = PIL.Image.fromarray(np_img).resize((640, 480)).rotate(180)

        t0 = perf_counter()
        # real_prediction = get_predictions_detectron(model, np.array(pil_img))
        # real_prediction = get_predictions_maskrcnn(model, np.array(pil_img))
        real_prediction = get_predictions_yolo(model, np.array(pil_img))
        dt = perf_counter() - t0
        print(f"Prediction took {dt:.2f} seconds")

        text_and_vertices = predictions_to_text_and_vertices(real_prediction)
        annotated = annotate(np.array(pil_img), text_and_vertices)

        cv2.imshow('annotated', annotated)

        k = cv2.waitKey(10)
        try:
            c = chr(k)
            if c == 's':
                save_idx = 0
                while True:
                    save_path = save_dir / f'img{save_idx}.png'
                    if save_path.exists():
                        save_idx += 1
                    else:
                        print(f"Saving {save_path.as_posix()}")
                        pil_img.save(save_path)
                        break
        except ValueError:
            pass


if __name__ == '__main__':
    main()
