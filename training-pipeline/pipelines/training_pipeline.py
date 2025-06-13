from kfp import dsl

from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component
from google_cloud_pipeline_components.v1.vertex_notification_email import VertexNotificationEmailOp

from components.preprocess_component import preprocess_component as PreprocessOp
from components.dataloader_component import dataloader_component as DataloaderOp
from components.hpt_component import hpt_component as HptOp
from components.training_component import training_component as TrainingOp
from components.deploy_component import deploy_component as DeployOp

import configuration.pipeline_config as pipeline_config

#wrapping components into Vertex AI custom training jobs
preprocessOP = create_custom_training_job_from_component(
    PreprocessOp,
    display_name = pipeline_config.DisplayNames.PREPROCESS_DISPLAY_NAME,
    machine_type = pipeline_config.ComputeResources.PREPROCESS_MACHINE_TYPE,
    service_account = pipeline_config.ServiceAccount.SERVICE_ACCOUNT
)

dataloaderOP = create_custom_training_job_from_component(
    DataloaderOp,
    display_name = pipeline_config.DisplayNames.DATALOADER_DISPLAY_NAME,
    machine_type = pipeline_config.ComputeResources.DATALOADER_MACHINE_TYPE,
    service_account = pipeline_config.ServiceAccount.SERVICE_ACCOUNT
)

hptOP = create_custom_training_job_from_component(
    HptOp,
    display_name = pipeline_config.DisplayNames.HPT_DISPLAY_NAME,
    machine_type = pipeline_config.ComputeResources.HPT_MACHINE_TYPE,
    service_account = pipeline_config.ServiceAccount.SERVICE_ACCOUNT
)

trainingOP = create_custom_training_job_from_component(
    TrainingOp,
    display_name = pipeline_config.DisplayNames.TRAINING_DISPLAY_NAME,
    machine_type = pipeline_config.ComputeResources.TRAINING_MACHINE_TYPE,
    service_account = pipeline_config.ServiceAccount.SERVICE_ACCOUNT
)

deployOP = create_custom_training_job_from_component(
    DeployOp,
    display_name = pipeline_config.DisplayNames.DEPLOY_DISPLAY_NAME,
    machine_type = pipeline_config.ComputeResources.DEPLOY_MACHINE_TYPE,
    service_account = pipeline_config.ServiceAccount.SERVICE_ACCOUNT
)

@dsl.pipeline(
    name = pipeline_config.Root.DISPLAY_NAME,
    description = pipeline_config.Root.DESCRIPTION
)
def training_pipeline(
    vertex_run_name: str,
    do_deploy: bool
):
    """Training pipeline"""
    
    notify_email_task = VertexNotificationEmailOp(
        recipients = pipeline_config.NotificationEmail.RECIPIENTS_LIST
    )
    with dsl.ExitHandler(
        exit_task = notify_email_task,
        name = 'Email Notification Exit Handler'
    ):
        #pipeline components
        preprocess_task = preprocessOP(
            project = pipeline_config.ProjectConfig.PROJECT_ID,
            location = pipeline_config.ProjectConfig.LOCATION,
            artifact_bucket = pipeline_config.ProjectConfig.ARTIFACT_BUCKET,
            data_bucket = pipeline_config.ProjectConfig.DATA_BUCKET,
            data_path = pipeline_config.ProjectConfig.DATA_PATH,
            vertex_experiment_name = pipeline_config.ExperimentConfig.EXPERIMENT_NAME,
            vertex_run_name = vertex_run_name
        )
        preprocess_task.set_display_name(pipeline_config.DisplayNames.PREPROCESS_DISPLAY_NAME)

        dataloader_task = dataloaderOP(
            project = pipeline_config.ProjectConfig.PROJECT_ID,
            location = pipeline_config.ProjectConfig.LOCATION,
            artifact_bucket = pipeline_config.ProjectConfig.ARTIFACT_BUCKET,
            vertex_experiment_name = pipeline_config.ExperimentConfig.EXPERIMENT_NAME,
            vertex_run_name = vertex_run_name,
            preprocessed_data_input = preprocess_task.outputs['preprocessed_data'],
        )
        dataloader_task.set_display_name(pipeline_config.DisplayNames.DATALOADER_DISPLAY_NAME)

        hpt_task = hptOP(
            project = pipeline_config.ProjectConfig.PROJECT_ID,
            location = pipeline_config.ProjectConfig.LOCATION,
            artifact_bucket = pipeline_config.ProjectConfig.ARTIFACT_BUCKET,
            vertex_experiment_name = pipeline_config.ExperimentConfig.EXPERIMENT_NAME,
            vertex_run_name = vertex_run_name,
            train_loader_input = dataloader_task.outputs['train_loader_output'],
            val_loader_input = dataloader_task.outputs['val_loader_output'],
        )
        hpt_task.set_display_name(pipeline_config.DisplayNames.HPT_DISPLAY_NAME)

        training_task = trainingOP(
            project = pipeline_config.ProjectConfig.PROJECT_ID,
            location = pipeline_config.ProjectConfig.LOCATION,
            artifact_bucket = pipeline_config.ProjectConfig.ARTIFACT_BUCKET,
            vertex_experiment_name = pipeline_config.ExperimentConfig.EXPERIMENT_NAME,
            vertex_run_name = vertex_run_name,
            training_input = dataloader_task.outputs['training_output'],
            train_loader_input = dataloader_task.outputs['train_loader_output'],
            val_loader_input = dataloader_task.outputs['val_loader_output'],
            best_params_input = hpt_task.outputs['best_params_output'],
        )
        training_task.set_display_name(pipeline_config.DisplayNames.TRAINING_DISPLAY_NAME)

        with dsl.If(
            do_deploy == True,
            name = 'Deploy Condition'
        ):
            deploy_task = deployOP(
                project = pipeline_config.ProjectConfig.PROJECT_ID,
                location = pipeline_config.ProjectConfig.LOCATION,
                artifact_bucket = pipeline_config.ProjectConfig.ARTIFACT_BUCKET,
                vertex_experiment_name = pipeline_config.ExperimentConfig.EXPERIMENT_NAME,
                vertex_run_name = vertex_run_name,
            )
            deploy_task.set_display_name(pipeline_config.DisplayNames.DEPLOY_DISPLAY_NAME)
            deploy_task.after(training_task)