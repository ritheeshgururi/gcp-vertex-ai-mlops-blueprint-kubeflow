from utils.utils import get_base_image_path

BASE_IMAGE_PATH = get_base_image_path()

class Root:
    DISPLAY_NAME = 'timeseries-tft-training-pipeline'
    PIPELINE_ROOT = f'gs://vertex-pipeline-root-training'
    PIPELINE_PACKAGE_YAML_PATH = 'compiled_pipeline_yaml/training_pipeline.yaml'
    DESCRIPTION = 'End to end pipeline for training the Temporal Fusion Transformer model'

class ProjectConfig:
    PROJECT_ID = 'gcp-vertexai-mlops-blueprint'
    LOCATION = 'asia-south1'
    ARTIFACT_BUCKET = 'training-pipeline-output-artifacts'
    DATA_BUCKET = 'volume-forecasting-data'
    DATA_PATH = 'original_merged_dataset.csv'
    
class ExperimentConfig:
    EXPERIMENT_NAME = 'tft-training-experiment-1'

class ControlFlow:
    DO_DEPLOY = False
    
class Dependencies:
    PREPROCESS_PACKAGES = [
        'pandas',
        'google-cloud-storage',
        'numpy',
        'google-cloud-aiplatform',
        'gcsfs'
    ]
    DATALOADER_PACKAGES = [
        'pandas',
        'numpy',
        'pytorch_forecasting',
        'torch',
        'google-cloud-storage',
        'google-cloud-aiplatform'
    ]
    HPT_PACKAGES = [
        'pytorch_forecasting',
        'torch',
        'lightning',
        'optuna',
        'statsmodels',
        'optuna-integration[pytorch_lightning]',
        'tensorboard',
        'google-cloud-storage',
        'google-cloud-aiplatform'
    ]
    TRAINING_PACKAGES = [
        'pytorch_forecasting',
        'torch',
        'lightning',
        'tensorboard',
        'pytorch_optimizer',
        'google-cloud-storage',
        'google-cloud-aiplatform'
    ]
    DEPLOY_PACKAGES = [
        'pandas',
        'google-cloud-storage',
        'google-cloud-aiplatform',
        'fastapi',
        # 'google-cloud-aiplatform[prediction]>=1.16.0'
    ]
    
class BaseImages:
    MACHINE_BASE_IMAGE = f'{BASE_IMAGE_PATH}'
    
# class TargetImages:
#     PREPROCESS_IMAGE = 'asia-south1-docker.pkg.dev/gcp-vertexai-mlops-blueprint/gcp-vertex-ai-mlops-blueprint/target_image:v10'
#     DATALOADER_IMAGE = 'asia-south1-docker.pkg.dev/gcp-vertexai-mlops-blueprint/gcp-vertex-ai-mlops-blueprint/dataloader_component'
#     HPT_IMAGE = 'asia-south1-docker.pkg.dev/gcp-vertexai-mlops-blueprint/gcp-vertex-ai-mlops-blueprint/hpt_component'
#     TRAINING_IMAGE = 'asia-south1-docker.pkg.dev/gcp-vertexai-mlops-blueprint/gcp-vertex-ai-mlops-blueprint/training_component'
    
class ComputeResources:
    PREPROCESS_MACHINE_TYPE = 'e2-standard-4'
    DATALOADER_MACHINE_TYPE = 'e2-standard-4'
    HPT_MACHINE_TYPE = 'e2-standard-4'
    TRAINING_MACHINE_TYPE = 'e2-standard-4'
    DEPLOY_MACHINE_TYPE = 'e2-standard-4'
    
class NotificationEmail:
    RECIPIENTS_LIST = ['ritheeshgururi187@gmail.com']
    
class ServiceAccount:
    SERVICE_ACCOUNT = 'gcp-vertexai-mlops-blueprint@gcp-vertexai-mlops-blueprint.iam.gserviceaccount.com'

class DisplayNames:
    PREPROCESS_DISPLAY_NAME = 'PreProcessing Pipeline'
    DATALOADER_DISPLAY_NAME = 'DataLoader Pipeline'
    HPT_DISPLAY_NAME = 'HPT Pipeline'
    TRAINING_DISPLAY_NAME = 'Training Pipeline'
    DEPLOY_DISPLAY_NAME = 'Deploy and Monitor Pipeline'