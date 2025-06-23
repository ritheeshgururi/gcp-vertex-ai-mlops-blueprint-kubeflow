# gcp-vertex-ai-mlops-blueprint

This repository provides an end-to-end MLOps workflow blueprint on GCP Vertex AI using Kubeflow. This comprises of the training, serving, batch inference, and online prediction of a Temporal Fusion Transformer model for time series forecasting.

The TFT model is trained on the open-source [Volume Forecasting dataset](https://www.kaggle.com/datasets/utathya/future-volume-forecasting), using this [Pytorch Forecasting](https://pytorch-forecasting.readthedocs.io/en/stable/) guide.

This README only details the setup instructions and steps to execute training and inference runs. For a detailed walkthrough of the code and the architectural decisions in this project, please refer to this Medium article: **[link]**

## Project Structure

There are four main directories in this project, each responsible for a specific stage of the MLOps lifecycle:

-   `training-pipeline/`: Contains a Kubeflow pipeline for model training and deployment, to be executed on Vertex Pipelines.
-   `serving-container-cpr/`: A model serving container built using Vertex AI Custom Prediction Routines.
-   `batch-inference-pipeline/`: Contains a Kubeflow pipeline to run Vertex AI batch prediction job.
-   `endpoint-prediction/`: A sample script to perform online predictions using a deployed model endpoint.

## Prerequisites

The execution environment should have the following installed and configured:
-   Python 3.11
-   Docker
-   A GCP project with all necessary APIs enabled.
-   Authenticated `gcloud` CLI.
-   All packages in the `requirements.txt` file. Run:
    ```shell
    pip install -r requirements.txt
    ```

## Setup and Execution Steps

Steps to configure and run the complete MLOps workflow.

### 1. Build the Serving Container
#### **Configuration**

Modify the follwing config variables in `serving-container-cpr/helper.py`:

-   `PROJECT_ID`: Project id of the project in which the artifact registry repository resides
-   `REGION`: GCP region in which the artifact registry repository resides
-   `REPOSITORY`: Name of the artifact registry repository to be used to store the serving container image
-   `IMAGE`: Name of the image with tag, to be used for pushing the serving container image

#### **Execution**

From the `serving-container-cpr` directory, run helper.py:

```shell
cd serving-container-cpr
python3 helper.py
```

---

### 2. Run the Training Pipeline
#### **Configuration**

Upload the `training-dataset/original_merged_dataset.csv` file to a GCS bucket in a location of your choice

Modify the following config variables in the files in `training-pipeline/configuration/`:

**File `step_config.py`:**

-   **class `DeployConfig`:**
    -   `upload_model_as_version`: Set to `True` if you want to upload the trained model as a version of an existing `Model` resource in Model Registry. Set to `False` if you want to register it as a standalone model
    -   `PARENT_MODEL`: The model ID or resource name of the parent model in the Model Registry, if the above is set to `True`
    -   `SERVING_CONTAINER_IMAGE_URI`: URI of the serving container image in Artifact Registry, as pushed in the previous step
    -   `deploy_to_existing_endpoint`: set to `True` if you want to deploy the model to an existing endpoint for online prediction, set to `False` if you want to create a new endpoint
    -   `EXISTING_ENDPOINT_ID`: The endpoint ID or resource name of the Vertex AI endpoint to deploy the model to, if the above is set to `True`
    -   `ENDPOINT_MACHINE_TYPE`: Machine configuration to be used for the endpoint

**File `pipeline_config.py`:**

-   **class `Root`:**
    -   `PIPELINE_ROOT`: GCS path to the directory to be used as the KFP pipeline root of the training pipeline
-   **class `ProjectConfig`:**
    -   `PROJECT_ID`: Project id of the GCP project to execute training pipeline in
    -   `LOCATION`: GCP region to execute pipeline run in
    -   `ARTIFACT_BUCKET`: GCS bucket name of the bucket to be used for archiving pipeline component artifacts
    -   `DATA_BUCKET`: GCS bucket name of the bucket in which training dataset is residing
    -   `DATA_PATH`: GCS path of the training dataset relative to the data bucket
-   **class `ExperimentConfig`:**
    -   `EXPERIMENT_NAME`: Name of the vertex experiment to be used for experiment tracking
-   **class `ControlFlow`:**
    -   `DO_DEPLOY`: Set to `True` to execute the deployment component; `False` to skip it
-   **class `ServiceAccount`:**
    -   `SERVICE_ACCOUNT`: The service account email to be used for authenticating the pipeline run

#### **Execution**

1.  **Create an Artifact Registry repository configured for docker in your preferred GCP region.**
2. **Authenticate**:
    ```shell
    gcloud auth configure-docker \
        <gcp-region>-docker.pkg.dev
    ```

3. **Build and push the base image for the KFP components:**
    *   From the `training-pipeline` directory, build the image:
        ```shell
        cd training-pipeline
        docker build -t <gcp-region>-docker.pkg.dev/<project-id>/<repository-name>/<base-image-name>:<tag> .
        ```
    *   Push the image to Artifact Registry:
        ```shell
        docker push <gcp-region>-docker.pkg.dev/<project-id>/<repository-name>/<base-image-name>:<tag>
        ```
    *   Optionally, to inspect or access the file system of the base image container interactively, run:
        ```bash
        docker run -it <gcp-region>-docker.pkg.dev/<project-id>/<repository-name>/<base-image-name>:<tag> /bin/bash
        ```
        **Example:**
        ```shell
        gcloud auth configure-docker \
        asia-south1-docker.pkg.dev
        docker build -t asia-south1-docker.pkg.dev/gcp-vertexai-mlops-blueprint/gcp-vertex-ai-mlops-blueprint/my-base-image:v39 .
        docker push asia-south1-docker.pkg.dev/gcp-vertexai-mlops-blueprint/gcp-vertex-ai-mlops-blueprint/my-base-image:v39
        ```

4.  **Start a pipeline run:**
    From the `training-pipeline` directory, run `main.py` with the full URI of the base image you just pushed as an argument.

    ```shell
    cd training-pipeline
    python3 main.py <gcp-region>-docker.pkg.dev/<project-id>/<repository-name>/<base-image-name>:<tag>
    ```
    **Example:**
    ```shell
    python3 main.py asia-south1-docker.pkg.dev/gcp-vertexai-mlops-blueprint/gcp-vertex-ai-mlops-blueprint/my-base-image:v39
    ```
---

### 3. Online Prediction
#### **Configuration**

Modify the following config variables in `endpoint-prediction/endpoint_prediction.py`:

-   `PROJECT_ID`: Project id of the GCP project in which the Vertex AI endpoint exists
-   `LOCATION`: GCP region in which the Vertex AI endpoint exists
-   `GCS_BUCKET`: GCS bucket in which the prediction csv file resides
-   `DATA_PATH`: The path to the prediction CSV file relative to `GCS_BUCKET`.
-   `ENDPOINT_ID`: Endpoint ID or the resource name of the Vertex AI endpoint to send the prediction request to

#### **Execution**

From `endpoint-prediction/`, run endpoint_predicttion.py:

```shell
cd endpoint-prediction
python3 endpoint-prediction/endpoint_prediction.py
```

---

### 4. Run the Batch Inference Pipeline
#### **Configuration**

Modify the following config variables in `batch-inference-pipeline/configurations/`:

File `step_config.py`:

-   class `BatchPredictionJobConfig`:
    -   `PROJECT_ID`: Project id of the project in which the Vertex AI batch prediction job is to be run
    -   `LOCATION`: GCP region in which the vertex AI batch prediction job is to be run
    -   `INPUT_URI`: GCS path to the .jsonl file to be sent as prediction input to the batch prediction job
    -   `OUTPUT_URI`: The GCS path to a directory where prediction results will be saved
    -   `MODEL_RESOURCE_NAME`: Model id or resource name of the Vertex AI Model in Model registry
    -   `DEPLOY_COMPUTE`: Machine configuration to be used for the Vertex AI batch prediction job

File `pipeline_config.py`:

-   class `Root`:
    -   `PIPELINE_ROOT`: GCS path to the directory to be used as the KFP pipeline root of the batch inference pipeline
    -   `SERVICE_ACCOUNT`: The service account email for authenticating the pipeline run.
-   class `ProjectConfig`:
    -   `PROJECT_ID`: Project id of the GCP project to execute training pipeline in
    -   `LOCATION`: The GCP region for the pipeline run.
-   class `ComputeResources`:
    -   `MACHINE_TYPE`: Machine configuration to be used for component execution of the batch inference component
-   class `NotificationEmail`:
    -   `NOTIFICATION_EMAIL_LIST`: A list of email recipients for pipeline status notifications.

#### **Execution**
Similar to the training pipeline,
1.  **Create an Artifact Registry repository configured for docker in your preferred GCP region.** 
2. **Build and push the base image for the KFP components:**

    *   From the `batch-inference-pipeline` directory, build the image:
        ```shell
        cd batch-inference-pipeline
        docker build -t <gcp-region>-docker.pkg.dev/<project-id>/<repository-name>/<batch-base-image-name>:<tag> .
        ```
    *   Push the image to Artifact Registry:
        ```shell
        docker push <gcp-region>-docker.pkg.dev/<project-id>/<repository-name>/<batch-base-image-name>:<tag>
        ```
3.  **Start a Pipeline Run:**
    From the `batch-inference-pipeline` directory, run `main.py` with the full URI of the base image you just pushed as an argument.
    ```shell
    batch-inference-pipeline
    python3 main.py <gcp-region>-docker.pkg.dev/<project-id>/<repository-name>/<batch-base-image-name>:<tag>
    ```
---
## IAM Permissions

This project was developed using impersonated service account credentials with broad permissions (`Owner` role). For environments that follow the principle of least privilege, grant the service account used for pipeline execution the necessary IAM roles for the following services:

-   **Vertex AI**
-   **Cloud Storage**
-   **Artifact Registry**
-   **Service Accounts**
-   **Compute Engine**
-   **Logging**