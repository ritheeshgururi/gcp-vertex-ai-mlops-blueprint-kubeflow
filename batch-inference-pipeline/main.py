from kfp.compiler import Compiler
import google.cloud.aiplatform as aiplatform
import configurations.pipeline_config as pipeline_config
from pipelines.batch_inference_pipeline import batch_inference_pipeline

#compiling the pipeline
compiler = Compiler()
compiler.compile(
    pipeline_func = batch_inference_pipeline,
    package_path = pipeline_config.Root.PIPELINE_PACKAGE_YAML_PATH
)

#initializing VertexAI
aiplatform.init(
    project = pipeline_config.ProjectConfig.PROJECT_ID,
    location = pipeline_config.ProjectConfig.LOCATION
)

#creating training pipeline job
batch_inference_job = aiplatform.PipelineJob(
    display_name = pipeline_config.Root.DISPLAY_NAME,
    template_path = pipeline_config.Root.PIPELINE_PACKAGE_YAML_PATH,
    pipeline_root = pipeline_config.Root.PIPELINE_ROOT,
    parameter_values = {
        'project': pipeline_config.ProjectConfig.PROJECT_ID,
        'location': pipeline_config.ProjectConfig.LOCATION,
        'gcs_bucket': pipeline_config.ProjectConfig.BUCKET_NAME,
        'data_path': pipeline_config.ProjectConfig.DATA_PATH,
        'endpoint_id': pipeline_config.ProjectConfig.ENDPOINT_ID
    },
    enable_caching = False
)

if __name__ == '__main__':
    batch_inference_job.submit()