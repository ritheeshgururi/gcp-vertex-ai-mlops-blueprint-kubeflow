from kfp import dsl
import configuration.pipeline_config as pipeline_config

@dsl.component(
    base_image = pipeline_config.BaseImages.MACHINE_BASE_IMAGE,
    packages_to_install = pipeline_config.Dependencies.HPT_PACKAGES,
    #uncomment below line to use target image if building base image with kfp component build
    # target_image = pipeline_config.TargetImages.HPT_IMAGE
)
def hpt_component(
    project: str,
    location: str,
    artifact_bucket: str,
    vertex_experiment_name: str,
    vertex_run_name: str,
    train_loader_input: dsl.Input[dsl.Dataset],
    val_loader_input: dsl.Input[dsl.Dataset],
    best_params_output: dsl.Output[dsl.Artifact]
):
    from steps.hpt_step import hpt_step
    
    hpt_step(
        project = project,
        location = location,
        artifact_bucket = artifact_bucket,
        vertex_experiment_name = vertex_experiment_name,
        vertex_run_name = vertex_run_name,
        train_loader_input = train_loader_input.path,
        val_loader_input = val_loader_input.path,
        best_params_output = best_params_output.path
    )