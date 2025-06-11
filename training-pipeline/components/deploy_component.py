from kfp import dsl
import configuration.pipeline_config as pipeline_config

@dsl.component(
    base_image = pipeline_config.BaseImages.MACHINE_BASE_IMAGE,
    packages_to_install = pipeline_config.Dependencies.DEPLOY_PACKAGES,
)
def deploy_component(
    project: str,
    location: str,
    artifact_bucket: str,
    vertex_run_name: str

):
    from steps.deploy_step import deploy_step
    
    deploy_step(
        project = project,
        location = location,
        artifact_bucket = artifact_bucket,
        vertex_run_name = vertex_run_name
    )