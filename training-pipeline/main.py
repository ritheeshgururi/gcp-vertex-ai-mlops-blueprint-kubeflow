from kfp.compiler import Compiler
from pipelines.training_pipeline import training_pipeline
import google.cloud.aiplatform as aiplatform
import configuration.pipeline_config as pipeline_config
import configuration.step_config as step_config
import datetime

#compiling the pipeline
compiler = Compiler()
compiler.compile(
    pipeline_func = training_pipeline,
    package_path = pipeline_config.Root.PIPELINE_PACKAGE_YAML_PATH
)

#initializing VertexAI with project id, location and experiment name
aiplatform.init(
    project = pipeline_config.ProjectConfig.PROJECT_ID,
    location = pipeline_config.ProjectConfig.LOCATION,
    experiment = pipeline_config.ExperimentConfig.EXPERIMENT_NAME
)

run_name = f'run-{(datetime.datetime.utcnow() + datetime.timedelta(hours=5, minutes=30)).strftime('%d%b%Y-%H-%M-%S').lower()}'

#associating the pipeline run with the unique experiment run name generated above
with aiplatform.start_run(run_name) as run:

    #creating training pipeline job
    training_pipeline_job = aiplatform.PipelineJob(
        display_name=pipeline_config.Root.DISPLAY_NAME,
        template_path=pipeline_config.Root.PIPELINE_PACKAGE_YAML_PATH,
        pipeline_root=pipeline_config.Root.PIPELINE_ROOT,
        parameter_values={
            'project': pipeline_config.ProjectConfig.PROJECT_ID,
            'location': pipeline_config.ProjectConfig.LOCATION,
            'vertex_experiment_name': pipeline_config.ExperimentConfig.EXPERIMENT_NAME,
            'vertex_run_name': run_name,
        },
        enable_caching = True,
    )

    #run/submit the job
    training_pipeline_job.run(service_account = pipeline_config.ServiceAccount.SERVICE_ACCOUNT)

    print(f'Pipeline submitted for run: {run_name}')

print(f'Run '{run_name}' completed.')