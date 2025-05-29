import os
from src_dir_handler_sdk.handler import CprHandler  
from src_dir_handler_sdk.predictor import CprPredictor
from google.cloud.aiplatform.prediction import LocalModel




def main():
    
    USER_SRC_DIR = "src_dir_handler_sdk"


    BUCKET_URI = "gs://training-data"
    MODEL_ARTIFACT_DIR = "model_artifacts"

    PROJECT_ID = "gcp-vertexai-mlops-blueprint"
    REGION = "us-central1"
    REPOSITORY = "mlops-repo"
    IMAGE = "inference-script-image-latest"


    MODEL_DISPLAY_NAME = "pytorch-tft-inference"
    print("Local Model Build Start")
   

    local_model = LocalModel.build_cpr_model(USER_SRC_DIR,
                                             f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{IMAGE}",
                                             predictor = CprPredictor,  
                                             handler = CprHandler,  
                                             requirements_path = os.path.join(USER_SRC_DIR, "requirements.txt"),
                                             base_image = 'python:3.10',
                                             no_cache = True)
    print("Local Model Build complete")

    local_model.push_image()
    print("Local Model Pushed to Registry")
    
    
if __name__ == "__main__":
    main()