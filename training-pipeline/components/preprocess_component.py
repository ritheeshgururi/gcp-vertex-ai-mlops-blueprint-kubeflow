from kfp import dsl
import configuration.pipeline_config as pipeline_config

@dsl.component(
    base_image = pipeline_config.BaseImages.MACHINE_BASE_IMAGE,
    packages_to_install = pipeline_config.Dependencies.PREPROCESS_PACKAGES,
    #uncomment below line to use target image if building base image with kfp component build
    # target_image = pipeline_config.TargetImages.PREPROCESS_IMAGE
)
def preprocess_component(
    project: str,
    location: str,
    artifact_bucket: str,
    data_bucket: str,
    data_path: str,
    vertex_experiment_name: str,
    vertex_run_name: str,
    preprocessed_data: dsl.Output[dsl.Dataset]
):
    from steps.preprocess_step import preprocess_step
    
    preprocess_step(
        project = project,
        location = location,
        artifact_bucket = artifact_bucket,
        data_bucket = data_bucket,
        data_path = data_path,
        vertex_experiment_name = vertex_experiment_name,
        vertex_run_name = vertex_run_name,
        preprocessed_data_path = preprocessed_data.path
    )