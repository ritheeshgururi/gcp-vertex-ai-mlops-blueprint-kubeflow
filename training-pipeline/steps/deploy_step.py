import logging

from google.cloud import aiplatform

from configuration.step_config import DeployConfig
from utils import step_utils

def deploy_step(
    project,
    location,
    artifact_bucket,
    vertex_experiment_name,
    vertex_run_name
):
    """Uploads the model to Model Registry and deploys it to an endpoint"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)    
    
    #getting experiment run name and initializing Vertex AI
    run = step_utils.get_vertex_experiment_run(
        project,
        location,
        vertex_experiment_name,
        vertex_run_name
    )

    logger.info('Uploading model resource to Model Registry')
    model = aiplatform.Model.upload(
        parent_model = DeployConfig.PARENT_MODEL if DeployConfig.deploy_as_version else None,
        display_name = DeployConfig.MODEL_DISPLAY_NAME,
        artifact_uri = f'gs://{artifact_bucket}/{vertex_run_name}/training_artifacts/ckpt_model_artifact/',
        serving_container_image_uri = DeployConfig.SERVING_CONTAINER_IMAGE_URI,
        serving_container_predict_route = DeployConfig.SERVING_CONTAINER_PREDICT_ROUTE,
        serving_container_health_route = DeployConfig.SERVING_CONTAINER_HEALTH_ROUTE
    )
    model.wait()
    logger.info('Model uploaded to Model Registry')

    run.log_params({
        'DEPLOY_OUTPUT_model_resource_name': model.resource_name
    })

    ###start of model deployment to endpoint
    if DeployConfig.deploy_to_existing_endpoint:
        #deploy to existing endpoint mode
        logger.info('Deploy to existing endpoint mode detected. Fetching existing endpoint')
        endpoint = aiplatform.Endpoint(
            endpoint_name = DeployConfig.EXISTING_ENDPOINT_ID
        )
        logger.info('Endpoint fetched succesfully')
    else:
        #deploy to new endpoint mode
        logger.info('Deploy to new endpoint mode detected. Start of endpoint creation')
        endpoint = aiplatform.Endpoint.create(
            display_name = DeployConfig.ENDPOINT_DISPLAY_NAME,
            project = project,
            location = location
        )
        logger.info('Endpoint created')

    logger.info('Start of deploying model to endpoint')
    deployed_model_endpoint = model.deploy(
        endpoint = endpoint,
        machine_type = DeployConfig.SERVING_CONTAINER_MACHINE_TYPE
    )
    logger.info('Model deployed to endpoint')
    ###end of model endpoint deployment

    run.log_params({
        'DEPLOY_OUTPUT_deployed_model_endpoint_resource_name': deployed_model_endpoint.resource_name
    })
    logger.info('Deploy component completed and artfacts logged to experiment run')
