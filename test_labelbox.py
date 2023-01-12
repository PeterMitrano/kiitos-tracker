import json
import os

import matplotlib.pyplot as plt
from labelbox.data.serialization import COCOConverter
from pycocotools.coco import COCO

from labelbox import Client

API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbGNtZ2I1OWgwMDN5MDcwcDM1NWJoNWdoIiwib3JnYW5pemF0aW9uSWQiOiJjbGNtZ2I1OTEwMDN4MDcwcDUxa3FlZnp1IiwiYXBpS2V5SWQiOiJjbGNta2E4azM1Ymd4MDd3Y2Vocmdjb28yIiwic2VjcmV0IjoiMjVmOTMwNTBiY2VmNTRkM2Y5YWEyNThhYWE4OGEwZmQiLCJpYXQiOjE2NzMxMzMxNTksImV4cCI6MjMwNDI4NTE1OX0.b95uFb5GosYK5g6GP0if_cxsqMqC_AD9RrXUtkBv7cY"


def main():
    client = Client(api_key=API_KEY)
    project = client.get_project('clcmgcpwe01my07yh0uc04shm')
    labels = project.label_generator()

    image_path = 'labelbox/images/'

    coco_labels = COCOConverter.serialize_instances(
        labels,
        image_root=image_path,
        ignore_existing_data=True
    )

    coco_labels['info']['image_root'] = coco_labels['info']['image_root'].as_posix()
    with open("labelbox_coco-1.json", 'w') as f:
        json.dump(coco_labels, f)


def show():
    coco = COCO("labelbox_coco-1.json")
    image_id = 5

    img_info = coco.imgs[image_id]

    cat_ids = coco.getCatIds()
    anns_ids = coco.getAnnIds(imgIds=img_info['id'], catIds=cat_ids, iscrowd=None)

    root = coco.dataset['info']['image_root']
    img_path = os.path.join(root, img_info['file_name'])
    img = plt.imread(img_path)
    plt.axis("off")
    plt.imshow(img)
    anns = coco.loadAnns(anns_ids)
    coco.showAnns(anns, draw_bbox=True)
    plt.savefig(f"annotated_by_pycoco.jpg", bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == '__main__':
    main()
    # show()
