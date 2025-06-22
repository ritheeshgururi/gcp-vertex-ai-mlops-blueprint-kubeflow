from utils import utils

#getting base image path as argument
BASE_IMAGE_PATH = utils.get_base_image_path()

class Root:
    DISPLAY_NAME = 'TFT Timeseries Training Pipeline'
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
    DO_DEPLOY = True
    
class Dependencies:
    PREPROCESS_PACKAGES = [
        'pandas == 2.3.0',
        'google-cloud-storage == 2.19.0',
        'numpy == 2.3.0',
        'google-cloud-aiplatform == 1.97.0',
        'gcsfs == 2025.5.1'
    ]
    DATALOADER_PACKAGES = [
        'pandas == 2.3.0',
        'numpy == 2.3.0',
        'pytorch_forecasting == 1.4.0',
        'torch == 2.7.1',
        'google-cloud-storage == 2.19.0',
        'google-cloud-aiplatform == 1.97.0'
    ]
    HPT_PACKAGES = [
        'pytorch_forecasting == 1.4.0',
        'torch == 2.7.1',
        'lightning == 2.5.1.post0',
        'optuna == 4.3.0',
        'statsmodels == 0.14.4',
        'optuna-integration[pytorch_lightning]',
        'tensorboard == 2.19.0',
        'google-cloud-storage == 2.19.0',
        'google-cloud-aiplatform == 1.97.0'
    ]
    TRAINING_PACKAGES = [
        'pytorch_forecasting == 1.4.0',
        'torch == 2.7.1',
        'lightning == 2.5.1.post0',
        'tensorboard == 2.19.0',
        'pytorch_optimizer',
        'google-cloud-storage == 2.19.0',
        'google-cloud-aiplatform == 1.97.0'
    ]
    DEPLOY_PACKAGES = [
        'pandas == 2.3.0',
        'google-cloud-storage == 2.19.0',
        'google-cloud-aiplatform == 1.97.0',
        'fastapi == 0.115.12'
    ]

class ExtraPipIndexUrls:
    DATALOADER_PIP_INDEX_URLS = ['https://download.pytorch.org/whl/cpu', 'https://pypi.org/simple']
    HPT_PIP_INDEX_URLS = ['https://download.pytorch.org/whl/cpu', 'https://pypi.org/simple']
    TRAINING_PIP_INDEX_URLS = ['https://download.pytorch.org/whl/cpu', 'https://pypi.org/simple']
        
class BaseImages:
    MACHINE_BASE_IMAGE = f'{BASE_IMAGE_PATH}'
    
#uncomment below lines to use target image if building base image with kfp build
# class TargetImages:
#     PREPROCESS_IMAGE = 'asia-south1-docker.pkg.dev/gcp-vertexai-mlops-blueprint/gcp-vertex-ai-mlops-blueprint/target_image:latest'
#     DATALOADER_IMAGE = 'asia-south1-docker.pkg.dev/gcp-vertexai-mlops-blueprint/gcp-vertex-ai-mlops-blueprint/dataloader_component:latest'
#     HPT_IMAGE = 'asia-south1-docker.pkg.dev/gcp-vertexai-mlops-blueprint/gcp-vertex-ai-mlops-blueprint/hpt_component:latest'
#     TRAINING_IMAGE = 'asia-south1-docker.pkg.dev/gcp-vertexai-mlops-blueprint/gcp-vertex-ai-mlops-blueprint/training_component:latest'
#     DEPLOY_IMAGE = 'asia-south1-docker.pkg.dev/gcp-vertexai-mlops-blueprint/gcp-vertex-ai-mlops-blueprint/training_component:latest'
    
class ComputeResources:
    PREPROCESS_MACHINE_TYPE = 'e2-standard-4'
    DATALOADER_MACHINE_TYPE = 'e2-standard-4'
    HPT_MACHINE_TYPE = 'e2-standard-4'
    TRAINING_MACHINE_TYPE = 'e2-standard-4'
    DEPLOY_MACHINE_TYPE = 'e2-standard-4'
    
class NotificationEmail:
    RECIPIENTS_LIST = ['ritheeshgururi187@gmail.com', 'gurugulapudi2024@gmail.com']
    
class ServiceAccount:
    SERVICE_ACCOUNT = 'gcp-vertexai-mlops-blueprint@gcp-vertexai-mlops-blueprint.iam.gserviceaccount.com'

class DisplayNames:
    PREPROCESS_DISPLAY_NAME = 'Preprocessing Component'
    DATALOADER_DISPLAY_NAME = 'Dataloader Component'
    HPT_DISPLAY_NAME = 'HPT Component'
    TRAINING_DISPLAY_NAME = 'Training Component'
    DEPLOY_DISPLAY_NAME = 'Deploy Component'