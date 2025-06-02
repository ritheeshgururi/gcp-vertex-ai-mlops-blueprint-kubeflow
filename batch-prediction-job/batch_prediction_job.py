import google.cloud.aiplatform as aiplatform
from config import BatchPredictionJobConfig, BatchPredictionMonitoringConfig

def batch_prediction_job():
    aiplatform.init(
        project = BatchPredictionJobConfig.PROJECT_ID,
        location = BatchPredictionJobConfig.LOCATION
    )

    # skew_config = aiplatform.model_monitoring.SkewDetectionConfig(
    #     data_source = BatchPredictionMonitoringConfig.TRAINING_DATASET_URI,
    #     skew_thresholds = BatchPredictionMonitoringConfig.SKEW_THRESHOLDS,
    #     attribute_skew_thresholds = BatchPredictionMonitoringConfig.ATTRIB_SKEW_THRESHOLDS,
    #     target_field = BatchPredictionMonitoringConfig.TARGET,
    #     data_format = BatchPredictionMonitoringConfig.TRAINING_DATASET_FORMAT
    # )

    # drift_config = aiplatform.model_monitoring.DriftDetectionConfig(
    #     drift_thresholds = BatchPredictionMonitoringConfig.DRIFT_THRESHOLDS,
    #     attribute_drift_thresholds = BatchPredictionMonitoringConfig.ATTRIB_DRIFT_THRESHOLDS,
    # )

    # objective_config = aiplatform.model_monitoring.ObjectiveConfig(
    #     skew_config, drift_config
    # )

    # alerting_config = aiplatform.model_monitoring.EmailAlertConfig(
    #     user_emails = BatchPredictionMonitoringConfig.NOTIFICATION_EMAILS,
    #     enable_logging = True
    # )

    batch_prediction_job = aiplatform.BatchPredictionJob.submit(
        job_display_name = BatchPredictionJobConfig.BATCH_PREDICTION_JOB_NAME,
        model_name = BatchPredictionJobConfig.MODEL_RESOURCE_NAME,
        instances_format = BatchPredictionJobConfig.REQUEST_FORMAT,
        predictions_format = BatchPredictionJobConfig.PREDICTIONS_FORMAT,
        gcs_source = BatchPredictionJobConfig.INPUT_URI,
        gcs_destination_prefix = BatchPredictionJobConfig.OUTPUT_URI,
        machine_type = BatchPredictionJobConfig.DEPLOY_COMPUTE,
        service_account = 'gcp-vertexai-mlops-blueprint@gcp-vertexai-mlops-blueprint.iam.gserviceaccount.com'
        # model_monitoring_objective_config = objective_config,
        # model_monitoring_alert_config = alerting_config,
    )

if __name__ == '__main__':
    batch_prediction_job()