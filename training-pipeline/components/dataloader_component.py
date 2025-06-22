from kfp import dsl
import configuration.pipeline_config as pipeline_config

@dsl.component(
    base_image = pipeline_config.BaseImages.MACHINE_BASE_IMAGE,
    packages_to_install = pipeline_config.Dependencies.DATALOADER_PACKAGES,
    pip_index_urls = pipeline_config.ExtraPipIndexUrls.DATALOADER_PIP_INDEX_URLS,
    #uncomment below line to use target image when using kfp component build
    # target_image = pipeline_config.TargetImages.DATALOADER_IMAGE
)
def dataloader_component(
    project: str,
    location: str,
    artifact_bucket: str,
    vertex_experiment_name: str,
    vertex_run_name: str,
    preprocessed_data_input: dsl.Input[dsl.Dataset],
    training_output: dsl.Output[dsl.Dataset],
    train_loader_output: dsl.Output[dsl.Dataset],
    val_loader_output: dsl.Output[dsl.Dataset]
): 
    from steps.dataloader_step import dataloader_step
    
    dataloader_step(
        project = project,
        location = location,
        artifact_bucket = artifact_bucket,
        vertex_experiment_name = vertex_experiment_name,
        vertex_run_name = vertex_run_name,
        preprocessed_data_input = preprocessed_data_input.path,
        training_output = training_output.path,
        train_loader_output = train_loader_output.path,
        val_loader_output = val_loader_output.path
    )