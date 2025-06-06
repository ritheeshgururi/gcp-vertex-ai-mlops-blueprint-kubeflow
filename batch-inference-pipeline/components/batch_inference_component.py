from kfp import dsl
import configurations.pipeline_config as pipeline_config

@dsl.component(
    base_image=pipeline_config.BaseImages.MACHINE_BASE_IMAGE,
    packages_to_install=pipeline_config.Dependencies.BATCH_PACKAGES,
)
def batch_inference_component(
    project: str,
    location: str,
    gcs_bucket: str,
    data_path: str,
    endpoint_id: str,
    predictions_output: dsl.Output[dsl.Dataset]
):
    from steps.batch_inference_step import batch_inference_step
    
    batch_inference_step(
        project = project,
        location = location,
        gcs_bucket = gcs_bucket,
        data_path = data_path,
        endpoint_id = endpoint_id,
        predictions_output = predictions_output.path
    )