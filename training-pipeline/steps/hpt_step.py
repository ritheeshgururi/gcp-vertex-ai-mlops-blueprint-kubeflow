from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from google.cloud import storage, aiplatform
import logging
from utils import step_utils
from configuration.step_config import HPTConfig

def hpt_step(
    project,
    location,
    artifact_bucket,
    vertex_experiment_name,
    vertex_run_name,
    train_loader_input,
    val_loader_input,
    best_params_output,
):
    """Tunes Hyperparameters using optuna"""
    
    ###fetching hyperparameter tuning variables/parameters from the step_config.py file
    n_trials = HPTConfig.n_trials
    max_epochs = HPTConfig.max_epochs
    gradient_clip_val_range = HPTConfig.gradient_clip_val_range
    hidden_size_range = HPTConfig.hidden_size_range
    hidden_continuous_size_range = HPTConfig.hidden_continuous_size_range
    attention_head_size_range = HPTConfig.attention_head_size_range
    learning_rate_range = HPTConfig.learning_rate_range
    dropout_range = HPTConfig.dropout_range
    trainer_kwargs_limit_train_batches = HPTConfig.trainer_kwargs_limit_train_batches
    reduce_on_plateau_patience = HPTConfig.reduce_on_plateau_patience
    use_learning_rate_finder = HPTConfig.use_learning_rate_finder
    
    ###logging parameters to the vertex experiment run
    aiplatform.init(project = project, location = location, experiment = vertex_experiment_name)
    run = aiplatform.ExperimentRun(run_name = vertex_run_name, experiment = vertex_experiment_name)
    run.log_params({
        'HPT_CONFIG_n_trials': n_trials,
        'HPT_CONFIG_max_epochs': max_epochs,
        'HPT_CONFIG_gradient_clip_val_range': str(gradient_clip_val_range),
        'HPT_CONFIG_hidden_size_range': str(hidden_size_range),
        'HPT_CONFIG_hidden_size_range': str(hidden_size_range),
        'HPT_CONFIG_hidden_continuous_size_range': str(hidden_continuous_size_range),
        'HPT_CONFIG_attention_head_size_range': str(attention_head_size_range),
        'HPT_CONFIG_learning_rate_range': str(learning_rate_range),
        'HPT_CONFIG_dropout_range': str(dropout_range),
        'HPT_CONFIG_trainer_kwargs_limit_train_batches': trainer_kwargs_limit_train_batches,
        'HPT_CONFIG_reduce_on_plateau_patience': reduce_on_plateau_patience,
        'HPT_CONFIG_use_learning_rate_finder': str(use_learning_rate_finder)
    })
    
    ###start of hyperparameter tuning logic
    logger = logging.getLogger(__name__)
    logger.info(f'Starting hyperparameter tuning')
    
    #loading data from component inputs    
    train_dataloader = step_utils.load_data_from_component_input_in_pickle(train_loader_input)
    val_dataloader = step_utils.load_data_from_component_input_in_pickle(val_loader_input)
    
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path = 'optuna_test',
        n_trials = n_trials,
        max_epochs = max_epochs,
        gradient_clip_val_range = gradient_clip_val_range,
        hidden_size_range = hidden_size_range,
        hidden_continuous_size_range = hidden_continuous_size_range,
        attention_head_size_range = attention_head_size_range,
        learning_rate_range = learning_rate_range,
        dropout_range = dropout_range,
        trainer_kwargs = dict(limit_train_batches = trainer_kwargs_limit_train_batches),
        reduce_on_plateau_patience = reduce_on_plateau_patience,
        use_learning_rate_finder = use_learning_rate_finder,
    )
    ###end of hyperparameter tuning logic
    
    #saving best trials to component output    
    step_utils.save_data_to_component_output_in_pickle(study.best_trial.params, best_params_output)
    
    #saving to GCS
    best_params_gcs_path = f'{vertex_run_name}/hpt_artifacts/best_params.pkl'
    step_utils.upload_file_to_gcs(project, artifact_bucket, best_params_output, best_params_gcs_path)
    
    logger.info(f'Best parameters saved to GCS: gs://{artifact_bucket}/{best_params_gcs_path}')