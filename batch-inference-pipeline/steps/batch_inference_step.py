import logging
from google.cloud import aiplatform
from configurations.step_config import BatchPredictionJobConfig
from configurations.step_config import BatchPredictionMonitoringConfig

def batch_inference_step():
    logger = logging.getLogger(__name__)

    logger.info('Starting batch inference job')
    batch_inference_job = aiplatform.BatchPredictionJob.submit(
        job_display_name = BatchPredictionJobConfig.BATCH_PREDICTION_JOB_NAME,
        model_name = BatchPredictionJobConfig.MODEL_RESOURCE_NAME,
        instances_format = BatchPredictionJobConfig.REQUEST_FORMAT,
        predictions_format = BatchPredictionJobConfig.PREDICTIONS_FORMAT,
        gcs_source = BatchPredictionJobConfig.INPUT_URI,
        gcs_destination_prefix = BatchPredictionJobConfig.OUTPUT_URI,
        machine_type = BatchPredictionJobConfig.DEPLOY_COMPUTE,
    )
    
    logger.info(f'Batch inference job: {batch_inference_job.resource_name} submitted successfully')