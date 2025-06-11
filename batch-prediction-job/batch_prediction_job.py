import google.cloud.aiplatform as aiplatform
from config import BatchPredictionJobConfig

def batch_prediction_job():
    aiplatform.init(
        project = BatchPredictionJobConfig.PROJECT_ID,
        location = BatchPredictionJobConfig.LOCATION
    )

    batch_prediction_job = aiplatform.BatchPredictionJob.submit(
        job_display_name = BatchPredictionJobConfig.BATCH_PREDICTION_JOB_NAME,
        model_name = BatchPredictionJobConfig.MODEL_RESOURCE_NAME,
        instances_format = BatchPredictionJobConfig.REQUEST_FORMAT,
        predictions_format = BatchPredictionJobConfig.PREDICTIONS_FORMAT,
        gcs_source = BatchPredictionJobConfig.INPUT_URI,
        gcs_destination_prefix = BatchPredictionJobConfig.OUTPUT_URI,
        machine_type = BatchPredictionJobConfig.DEPLOY_COMPUTE,
        service_account = 'gcp-vertexai-mlops-blueprint@gcp-vertexai-mlops-blueprint.iam.gserviceaccount.com'
    )

if __name__ == '__main__':
    batch_prediction_job()