import pathlib
from datetime import datetime

from PIL import Image
from PyQt5.QtCore import QObject

from kiitos.upload_for_labeling import make_labelbox_client, upload_image_to_bucket, upload_to_labelbox


class CaptureWorker(QObject):

    def __init__(self, capture_manager):
        super().__init__()
        self.capture_manager = capture_manager

    def save_last_frame(self):
        pil_img = Image.fromarray(self.capture_manager.last_frame)
        pil_img = pil_img.rotate(180)
        image_path = pathlib.Path(f'saved_from_live/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png')
        pil_img.save(image_path)
        labelbox_client = make_labelbox_client()
        url = upload_image_to_bucket(image_path)
        upload_to_labelbox(labelbox_client, url)
        print(f"Saved image to {image_path}, and uploaded it for labeling")
