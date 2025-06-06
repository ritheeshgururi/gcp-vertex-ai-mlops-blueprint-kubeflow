from kfp import dsl
import configurations.pipeline_config as pipeline_config

@dsl.component(
    base_image=pipeline_config.BaseImages.MACHINE_BASE_IMAGE,
    packages_to_install=pipeline_config.Dependencies.BATCH_PACKAGES,
)
def batch_inference_component():
    from steps.batch_inference_step import batch_inference_step
    
    batch_inference_step()