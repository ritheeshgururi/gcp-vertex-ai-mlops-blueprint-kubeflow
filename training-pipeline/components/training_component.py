from kfp import dsl
import configuration.pipeline_config as pipeline_config

@dsl.component(
    base_image = pipeline_config.BaseImages.MACHINE_BASE_IMAGE,
    packages_to_install = pipeline_config.Dependencies.TRAINING_PACKAGES,
    pip_index_urls = pipeline_config.EXTRA_PIP_INDEX_URLS.TRAINING_PIP_INDEX_URLS,
    #uncomment below line to use target image if building base image with kfp component build
    # target_image = pipeline_config.TargetImages.TRAINING_IMAGE
)
def training_component(
    project: str,
    location: str,
    artifact_bucket: str,
    vertex_experiment_name: str,
    vertex_run_name: str,
    training_input: dsl.Input[dsl.Dataset],
    train_loader_input: dsl.Input[dsl.Dataset],
    val_loader_input: dsl.Input[dsl.Dataset],
    best_params_input: dsl.Input[dsl.Artifact],
    pth_model_output: dsl.Output[dsl.Model],
    ckpt_model_output: dsl.Output[dsl.Model]
):
    from steps.training_step import training_step
    
    training_step(
        project = project,
        location = location,
        artifact_bucket = artifact_bucket,
        vertex_experiment_name = vertex_experiment_name,
        vertex_run_name = vertex_run_name,
        training_input = training_input.path,
        train_loader_input = train_loader_input.path,
        val_loader_input = val_loader_input.path,
        best_params_input = best_params_input.path,
        pth_model_output = pth_model_output.path,
        ckpt_model_output = ckpt_model_output.path
    )