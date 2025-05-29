import time
import google.cloud.aiplatform as aiplatform

def batch_prediction_job():

    PROJECT_ID = "gcp-vertexai-mlops-blueprint"
    LOCATION = "us-central1"
    BATCH_PREDICTION_JOB_NAME = "TFT Batch Predictions"
    TRAINING_DATASET_URI = "gs://training-data-/merged_dataset.csv"
    TRAINING_DATASET_FORMAT = "csv"
    TARGET = "volume"
    INPUT_URI = "gs://batch-prediction-data/input/test.jsonl"
    OUTPUT_URI = "gs://batch-prediction-data/output/"
    MODEL_NAME = "projects/123456789/locations/us-central1/models/0123456789"
    DEPLOY_COMPUTE = "n2-standard-16"

    DEFAULT_THRESHOLD_VALUE = 0.001

    SKEW_THRESHOLDS = {
        "volume": DEFAULT_THRESHOLD_VALUE,
    }
    DRIFT_THRESHOLDS = {
        "volume": DEFAULT_THRESHOLD_VALUE,
    }
    ATTRIB_SKEW_THRESHOLDS = {
        "industry_volume": DEFAULT_THRESHOLD_VALUE,
        "avg_volume_by_agency": DEFAULT_THRESHOLD_VALUE,
        "avg_volume_by_sku": DEFAULT_THRESHOLD_VALUE,
    }
    ATTRIB_DRIFT_THRESHOLDS = {
        "price_regular": DEFAULT_THRESHOLD_VALUE,
        "price_actual": DEFAULT_THRESHOLD_VALUE,
        "discount_in_percent": DEFAULT_THRESHOLD_VALUE,
    }


#     skew_config = aiplatform.model_monitoring.SkewDetectionConfig(
#         data_source=TRAINING_DATASET_URI,
#         skew_thresholds=SKEW_THRESHOLDS,
#         attribute_skew_thresholds=ATTRIB_SKEW_THRESHOLDS,
#         target_field=TARGET,
#         data_format=TRAINING_DATASET_FORMAT
#     )

#     drift_config = aiplatform.model_monitoring.DriftDetectionConfig(
#         drift_thresholds=DRIFT_THRESHOLDS,
#         attribute_drift_thresholds=ATTRIB_DRIFT_THRESHOLDS,
#     )

#     objective_config = aiplatform.model_monitoring.ObjectiveConfig(
#         skew_config, drift_config
#     )

    emails = ["ritheeshgururi187@gmail.com", "gurugulapudi2024@gmail.com"]
    alerting_config = aiplatform.model_monitoring.EmailAlertConfig(
        user_emails=emails, enable_logging=True
    )

    batch_prediction_job = aiplatform.BatchPredictionJob.submit(
        job_display_name=BATCH_PREDICTION_JOB_NAME,
        model_name=MODEL_NAME,
        instances_format="jsonl",
        predictions_format="jsonl",
        gcs_source=INPUT_URI,
        gcs_destination_prefix=OUTPUT_URI,
        machine_type=DEPLOY_COMPUTE,
        # model_monitoring_objective_config=objective_config,
        model_monitoring_alert_config=alerting_config,
    )

if __name__ == "__main__":
    batch_prediction_job()