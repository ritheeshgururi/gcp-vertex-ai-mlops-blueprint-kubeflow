from utils.utils import get_base_image_path

class Root:
    DISPLAY_NAME = "containerized-batch-prediction-pipeline"
    PIPELINE_ROOT = f"gs://vertex-batch-pipeline"
    PIPELINE_PACKAGE_YAML_PATH = "compiled_pipeline_yaml/batch_prediction_pipeline.yaml"
    DESCRIPTION = "End to end pipeline for batch prediction"

class ProjectConfig:
    PROJECT_ID = "gcp-vertexai-mlops-blueprint"
    LOCATION = "us-central1"
    BUCKET_NAME = "batch-prediction-data"
    DATA_PATH = "input/test_dataset.csv"
    ENDPOINT_ID = "projects/123456789/locations/us-central1/endpoints/0123456789"

class Dependencies:
    BATCH_PACKAGES = [
        "pandas",
        "google-cloud-storage",
        "google-cloud-aiplatform"
    ]

class BaseImages:
    MACHINE_BASE_IMAGE = get_base_image_path()

class ComputeResources:
    BATCH_MACHINE_TYPE = "n2-standard-4"

class NotificationEmail:
    NOTIFICATION_EMAIL_LIST = ["ritheeshgururi187@gmail.com"]
