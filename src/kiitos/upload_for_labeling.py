from google.cloud import storage
from labelbox import Client

API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbGNtZ2I1OWgwMDN5MDcwcDM1NWJoNWdoIiwib3JnYW5pemF0aW9uSWQiOiJjbGNtZ2I1OTEwMDN4MDcwcDUxa3FlZnp1IiwiYXBpS2V5SWQiOiJjbGNta2E4azM1Ymd4MDd3Y2Vocmdjb28yIiwic2VjcmV0IjoiMjVmOTMwNTBiY2VmNTRkM2Y5YWEyNThhYWE4OGEwZmQiLCJpYXQiOjE2NzMxMzMxNTksImV4cCI6MjMwNDI4NTE1OX0.b95uFb5GosYK5g6GP0if_cxsqMqC_AD9RrXUtkBv7cY"


def make_labelbox_client():
    client = Client(api_key=API_KEY)
    return client


def upload_image_to_bucket(image_path):
    bucket_name = "kiitos-saved-from-live"
    blob_name = image_path.name
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(image_path.as_posix())
    blob.make_public()
    url = f'https://storage.googleapis.com/{bucket_name}/{blob_name}'
    return url


def upload_to_labelbox(client, image_url):
    dataset = get_live_dataset(client)
    data_row = dataset.create_data_row(row_data=image_url, media_type="IMAGE")
    queue_for_labeling(client, data_row)


def queue_for_labeling(client, data_row):
    project = client.get_project("clcmgcpwe01my07yh0uc04shm")
    project.create_batch(f"batch-{data_row.uid}", [data_row], 1)


def get_live_dataset(client):
    DATASET_ID = "clcu1bq1w1wec07z1ekjjfyr4"
    dataset = client.get_dataset(DATASET_ID)
    return dataset
