import logging

from google.cloud import aiplatform

from configuration.step_config import DeployConfig

def deploy_step(
    project,
    location,
    artifact_bucket,
    vertex_run_name,
):
    """Uploads the model to Model Registry, deploys the same to an endpoint, and creates model monitoring job"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)    
    
    aiplatform.init(project = project, location = location)

    logger.info('Uploading model resource to Model Registry')
    model = aiplatform.Model.upload(
        parent_model = DeployConfig.PARENT_MODEL,
        display_name = DeployConfig.MODEL_DISPLAY_NAME,
        artifact_uri = f'gs://{artifact_bucket}/{vertex_run_name}/ckpt_model_artifacts',
        serving_container_image_uri = DeployConfig.SERVING_CONTAINER_IMAGE_URI,
        serving_container_predict_route = '/predict',
        serving_container_health_route = '/health'
    )
    logger.info('Model uploaded to Model Registry')

    ###start of model deployment to endpoint
    if DeployConfig.deploy_to_existing_endpoint == True:
        #deploy to existing endpoint mode
        logger.info('Deploy to existing endpoint mode detected. Fetching existing endpoint')
        endpoint = aiplatform.Endpoint(
            endpoint_name = DeployConfig.EXISTING_ENDPOINT_ID
        )
        logger.info('Endpoint fetched succesfully')
    else:
        #deploy to new endoint mode
        logger.info('Deploy to new endpoint mode detected. Start of endpoint creation')
        endpoint = aiplatform.Endpoint.create(
            display_name = DeployConfig.ENDPOINT_DISPLAY_NAME,
            project = project,
            location = location
        )
        logger.info('Endpoint created')

    logger.info('Start of deploying model to endpoint')
    endpoint = model.deploy(
        endpoint = endpoint,
        machine_type = DeployConfig.SERVING_CONTAINER_MACHINE_TYPE
    )
    logger.info('Model deployed to endpoint')
    ###end of model endpoint deployment