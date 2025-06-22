import logging
from google.cloud import aiplatform
from configurations.step_config import BatchPredictionJobConfig

def batch_inference_step():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    aiplatform.init(
        project = BatchPredictionJobConfig.PROJECT_ID,
        location = BatchPredictionJobConfig.LOCATION
    )

    logger.info('Instantiating Model Registry Model resource')
    model = aiplatform.Model(
        model_name = BatchPredictionJobConfig.MODEL_RESOURCE_NAME
    )
    logger.info('Model Registry Model resource fetched succesfully')

    logger.info('Starting batch inference job')
    batch_inference_job = model.batch_predict(
        job_display_name = BatchPredictionJobConfig.BATCH_PREDICTION_JOB_NAME,
        instances_format = BatchPredictionJobConfig.REQUEST_FORMAT,
        predictions_format = BatchPredictionJobConfig.PREDICTIONS_FORMAT,
        gcs_source = BatchPredictionJobConfig.INPUT_URI,
        gcs_destination_prefix = BatchPredictionJobConfig.OUTPUT_URI,
        machine_type = BatchPredictionJobConfig.DEPLOY_COMPUTE
    )
    logger.info('Batch prediction job submitted succesfully')

    logger.info(f'Batch inference job: {batch_inference_job.resource_name} submitted successfully')