from labelbox import Client

API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbGNtZ2I1OWgwMDN5MDcwcDM1NWJoNWdoIiwib3JnYW5pemF0aW9uSWQiOiJjbGNtZ2I1OTEwMDN4MDcwcDUxa3FlZnp1IiwiYXBpS2V5SWQiOiJjbGNta2E4azM1Ymd4MDd3Y2Vocmdjb28yIiwic2VjcmV0IjoiMjVmOTMwNTBiY2VmNTRkM2Y5YWEyNThhYWE4OGEwZmQiLCJpYXQiOjE2NzMxMzMxNTksImV4cCI6MjMwNDI4NTE1OX0.b95uFb5GosYK5g6GP0if_cxsqMqC_AD9RrXUtkBv7cY"


def make_labelbox_client():
    client = Client(api_key=API_KEY)
    return client


def upload_to_labelbox(client, image_path):
    dataset = get_live_dataset(client)
    data_row = dataset.create_data_row(row_data=image_path.as_posix())
    queue_for_labeling(client, data_row)


def queue_for_labeling(client, data_row):
    project = client.get_project("clcmgcpwe01my07yh0uc04shm")
    project.create_batch(f"batch-{data_row.uid}", [data_row], 1)


def get_live_dataset(client):
    DATASET_ID = "clcu1bq1w1wec07z1ekjjfyr4"
    dataset = client.get_dataset(DATASET_ID)
    return dataset
