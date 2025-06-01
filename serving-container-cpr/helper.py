import os
from src_dir_handler_sdk.handler import CprHandler  
from src_dir_handler_sdk.predictor import CprPredictor
from google.cloud.aiplatform.prediction import LocalModel

def main():
    USER_SRC_DIR = "src_dir_handler_sdk"
    PROJECT_ID = "gcp-vertexai-mlops-blueprint"
    REGION = "asia-south1"
    REPOSITORY = "tft-cpr-serving-container"
    IMAGE = "serving-container-cpr:latest"

    print("Building local model")

    local_model = LocalModel.build_cpr_model(
        USER_SRC_DIR,
        f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{IMAGE}",
        predictor = CprPredictor,  
        handler = CprHandler,  
        requirements_path = os.path.join(USER_SRC_DIR, "cpr_requirements.txt"),
        base_image = 'python:3.11-slim-bookworm',
        no_cache = True
    )
    print("Local model built succesfully")

    local_model.push_image()
    print("Prediction container image pushed to artifact registry")
    
if __name__ == "__main__":
    main()