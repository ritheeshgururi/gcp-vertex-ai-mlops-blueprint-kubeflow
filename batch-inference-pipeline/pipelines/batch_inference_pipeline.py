from kfp import dsl

from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component
from google_cloud_pipeline_components.v1.vertex_notification_email import VertexNotificationEmailOp

from components.batch_inference_component import batch_inference_component as BatchInferenceOp
import configurations.pipeline_config as pipeline_config

batchinferenceOP = create_custom_training_job_from_component(
    BatchInferenceOp,
    display_name = pipeline_config.DisplayNames.BATCH_INFERENCE_DISPLAY_NAME,
    machine_type = pipeline_config.ComputeResources.MACHINE_TYPE,
    service_account = pipeline_config.Root.SERVICE_ACCOUNT
)

@dsl.pipeline(
    name = pipeline_config.Root.DISPLAY_NAME,
    description = pipeline_config.Root.DESCRIPTION
)
def batch_inference_pipeline():
    notify_email_task = VertexNotificationEmailOp(recipients = pipeline_config.NotificationEmail.NOTIFICATION_EMAIL_LIST)
    
    with dsl.ExitHandler(
        notify_email_task,
        name = 'Email Notification Exit Handler'
    ):
        batch_inference_task = batchinferenceOP()
        batch_inference_task.set_display_name(pipeline_config.DisplayNames.BATCH_INFERENCE_DISPLAY_NAME)