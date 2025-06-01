import logging
from google.cloud import aiplatform
from configuration.step_config import DeployConfig, MonitorConfig

def deploy_and_monitor_step(
    project,
    location,
    artifact_bucket,
    vertex_run_name,
):
    """Deploys the model endpoint to model registry and creates model monitoring job"""
    logger = logging.getLogger(__name__)
    
    ###start of model endpoint deployment
    logger.info("Deploying Model to Model Registry")

    aiplatform.init(project = project, location = location)

    model = aiplatform.Model.upload(
        parent_model = DeployConfig.PARENT_MODEL,
        display_name = DeployConfig.MODEL_DISPLAY_NAME,
        artifact_uri = f"gs://{artifact_bucket}/{vertex_run_name}/ckpt_model_artifacts",
        serving_container_image_uri = DeployConfig.SERVING_CONTAINER_IMAGE_URI,
        serving_container_predict_route = '/predict',
        serving_container_health_route = '/health'
    )
    logger.info("Model uploaded to artifact registry")

    # endpoint = aiplatform.Endpoint.create(
    #     display_name = DeployConfig.MODEL_DISPLAY_NAME,
    #     project = project,
    #     location = location
    # )
    # logger.info("Endpoint created")

    # endpoint = model.deploy(
    #     endpoint = endpoint,
    #     machine_type = DeployConfig.SERVING_CONTAINER_MACHINE_TYPE
    # )
    # logger.info("Endpoint deployed")
    # ###end of model endpoint deployment
    
    # ###start of model monitoring job creation
    # logger.info("Model monintoring started")
    
    # skew_config = aiplatform.model_monitoring.SkewDetectionConfig(
    #     data_source = MonitorConfig.DATASET_URI,
    #     skew_thresholds = MonitorConfig.SKEW_THRESHOLDS,
    #     attribute_skew_thresholds = MonitorConfig.ATTRIB_SKEW_THRESHOLDS,
    #     target_field = MonitorConfig.TARGET,
    # )

    # drift_config = aiplatform.model_monitoring.DriftDetectionConfig(
    #     drift_thresholds = MonitorConfig.DRIFT_THRESHOLDS,
    #     attribute_drift_thresholds = MonitorConfig.ATTRIB_DRIFT_THRESHOLDS,
    # )

    # objective_config = aiplatform.model_monitoring.ObjectiveConfig(
    #     skew_config, drift_config
    # )
    
    # logger.info("Model monitoring configurations created")
        
    # #create sampling configuration
    # random_sampling = aiplatform.model_monitoring.RandomSampleConfig(sample_rate = MonitorConfig.LOG_SAMPLE_RATE)

    # #create schedule configuration
    # schedule_config = aiplatform.model_monitoring.ScheduleConfig(monitor_interval = MonitorConfig.MONITOR_INTERVAL)

    # #create alerting configuration.
    # emails = MonitorConfig.USER_EMAIL
    # alerting_config = aiplatform.model_monitoring.EmailAlertConfig(
    #     user_emails = emails, enable_logging = True
    # )
    
    
    # logger.info("Alerting configurations created")
    
    # # Create the monitoring job.
    # job = aiplatform.ModelDeploymentMonitoringJob.create(
    #     display_name = MonitorConfig.JOB_NAME,
    #     logging_sampling_strategy = random_sampling,
    #     schedule_config = schedule_config,
    #     alert_config = alerting_config,
    #     objective_configs = objective_config,
    #     project = project,
    #     location = location,
    #     endpoint = endpoint,
    # )
    
    # logger.info("Model monitoring completed")
    # ###end of model monitoring job creation
