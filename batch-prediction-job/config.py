class BatchPredictionJobConfig:
    PROJECT_ID = "gcp-vertexai-mlops-blueprint"
    LOCATION = "asia-south1"
    BATCH_PREDICTION_JOB_NAME = "TFT Batch Prediction"
    INPUT_URI = "gs://batch-prediction-volume-forecasting/input/input.jsonl"
    OUTPUT_URI = "gs://batch-prediction-volume-forecasting/output/"
    MODEL_RESOURCE_NAME = "projects/848202130152/locations/asia-south1/models/9077488834081456128"
    DEPLOY_COMPUTE = "n2-standard-2"
    REQUEST_FORMAT = "jsonl"
    PREDICTIONS_FORMAT = "jsonl"

class BatchPredictionMonitoringConfig:
    TRAINING_DATASET_URI = "gs://training-data-/merged_dataset.csv"
    DEFAULT_THRESHOLD_VALUE = 0.001
    TARGET = "volume"
    TRAINING_DATASET_FORMAT = "csv"

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

    NOTIFICATION_EMAILS = ["ritheeshgururi187@gmail.com", "gurugulapudi2024@gmail.com"]