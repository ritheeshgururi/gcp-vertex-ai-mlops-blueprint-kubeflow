from utils.utils import get_base_image_path

class Root:
    DISPLAY_NAME = 'TFT Batch Inference Pipeline'
    PIPELINE_ROOT = f'gs://vertex-pipeline-root-inference'
    PIPELINE_PACKAGE_YAML_PATH = 'compiled_pipeline_yaml/batch_inference_pipeline.yaml'
    DESCRIPTION = 'End to end pipeline for batch inference'
    SERVICE_ACCOUNT = 'gcp-vertexai-mlops-blueprint@gcp-vertexai-mlops-blueprint.iam.gserviceaccount.com'

class ProjectConfig:
    PROJECT_ID = 'gcp-vertexai-mlops-blueprint'
    LOCATION = 'asia-south1'

class Dependencies:
    BATCH_PACKAGES = [
        'pandas ==  2.3.0',
        'google-cloud-storage == 2.19.0',
        'google-cloud-aiplatform == 1.97.0'
    ]

class BaseImages:
    MACHINE_BASE_IMAGE = get_base_image_path()

class ComputeResources:
    MACHINE_TYPE = 'n2-standard-4'

class NotificationEmail:
    NOTIFICATION_EMAIL_LIST = ['ritheeshgururi187@gmail.com', 'gurugulapudi2024@gmail.com']

class DisplayNames:
    BATCH_INFERENCE_DISPLAY_NAME = 'Batch Inference Component'