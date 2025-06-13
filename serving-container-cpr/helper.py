import os

from google.cloud.aiplatform.prediction import LocalModel

from src_dir_handler_sdk.handler import CprHandler  
from src_dir_handler_sdk.predictor import CprPredictor

def main():
    USER_SRC_DIR = 'src_dir_handler_sdk'
    PROJECT_ID = 'gcp-vertexai-mlops-blueprint'
    REGION = 'asia-south1'
    REPOSITORY = 'tft-cpr-serving-container'
    IMAGE = 'serving-container-cpr:latest'
    BASE_IMAGE = 'python:3.11-slim-bookworm'

    print('Local Model Build Start')

    local_model = LocalModel.build_cpr_model(
        USER_SRC_DIR,
        f'{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{IMAGE}',
        predictor = CprPredictor,  
        handler = CprHandler,  
        requirements_path = os.path.join(USER_SRC_DIR, 'cpr_requirements.txt'),
        base_image = BASE_IMAGE,
        no_cache = True
    )
    print('Local Model Build complete')

    local_model.push_image()
    print('Local Model Image Pushed to Aritfact Registry')
    
if __name__ == '__main__':
    main()