from kfp import dsl

from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component
from google_cloud_pipeline_components.v1.vertex_notification_email import VertexNotificationEmailOp

from components.endpoint_prediction_component import endpoint_prediction_component as EndPointPredictionOp

import configurations.pipeline_config as pipeline_config

endpointpredictionOP = create_custom_training_job_from_component(
    EndPointPredictionOp,
    display_name = pipeline_config.Root.DISPLAY_NAME,
    machine_type = pipeline_config.ComputeResources.MACHINE_TYPE,
    service_account = pipeline_config.Root.SERVICE_ACCOUNT
)

@dsl.pipeline(
    name = pipeline_config.Root.DISPLAY_NAME,
    description = pipeline_config.Root.DESCRIPTION
)
def endpoint_inference_pipeline(
    project: str,
    location: str,
    gcs_bucket: str,
    data_path: str,
    endpoint_id: str
):
    notify_email_task = VertexNotificationEmailOp(recipients = pipeline_config.NotificationEmail.NOTIFICATION_EMAIL_LIST)
    
    with dsl.ExitHandler(notify_email_task):
        endpoint_prediction_task = endpointpredictionOP(
            project = project,
            location=location,
            gcs_bucket=gcs_bucket,
            data_path=data_path,
            endpoint_id=endpoint_id
        )