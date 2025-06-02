from utils.utils import get_base_image_path

class Root:
    DISPLAY_NAME = 'endoint-inference-pipeline'
    PIPELINE_ROOT = f'gs://vertex-pipeline-root-inference'
    PIPELINE_PACKAGE_YAML_PATH = 'compiled_pipeline_yaml/batch_prediction_pipeline.yaml'
    DESCRIPTION = 'End to end pipeline for endpoint prediction'
    SERVICE_ACCOUNT = 'gcp-vertexai-mlops-blueprint@gcp-vertexai-mlops-blueprint.iam.gserviceaccount.com'

class ProjectConfig:
    PROJECT_ID = 'gcp-vertexai-mlops-blueprint'
    LOCATION = 'asia-south1'
    BUCKET_NAME = 'online-prediction-volume-forecasting'
    DATA_PATH = 'input/endpoint_prediction_dataset.csv'
    ENDPOINT_ID = '1868373720800690176'

class Dependencies:
    BATCH_PACKAGES = [
        'pandas',
        'google-cloud-storage',
        'google-cloud-aiplatform'
    ]

class BaseImages:
    MACHINE_BASE_IMAGE = get_base_image_path()

class ComputeResources:
    MACHINE_TYPE = 'n2-standard-4'

class NotificationEmail:
    NOTIFICATION_EMAIL_LIST = ['ritheeshgururi187@gmail.com']