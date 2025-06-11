import logging
import pandas as pd

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

from configuration.step_config import PreprocessConfig, DataloaderConfig
from utils import step_utils

def dataloader_step(
    project,
    location,
    artifact_bucket,
    preprocessed_data_input,
    training_output,
    train_loader_output,
    val_loader_output,
    vertex_experiment_name,
    vertex_run_name,
):
    """Creates training and validation datasets."""

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    ###fetching the dataloader component variables/parameters from the step_config.py file
    special_days = PreprocessConfig.special_days
    batch_size = DataloaderConfig.batch_size
    max_prediction_length = DataloaderConfig.max_prediction_length
    max_encoder_length = DataloaderConfig.max_encoder_length
    min_max_encoder_length_ratio = DataloaderConfig.min_max_encoder_length_ratio
    min_prediction_length = DataloaderConfig.min_prediction_length
    target_normalizer_transformation = DataloaderConfig.target_normalizer_transformation
    num_workers = DataloaderConfig.num_workers
    train_val_batch_size_ratio = DataloaderConfig.train_val_batch_size_ratio
    
    ###logging parameters to the vertex experiment run
    logger.info('Initializing Vertex AI with experiment name and instantiating experiment run')
    run = step_utils.get_vertex_experiment_run(
        project,
        location,
        vertex_experiment_name,
        vertex_run_name
    )

    logger.info('Logging parameters to the experiment run')
    run.log_params({
        'DATALOADER_CONFIG_batch_size': batch_size,
        'DATALOADER_CONFIG_max_prediction_length': max_prediction_length,
        'DATALOADER_CONFIG_max_encoder_length': max_encoder_length,
        'DATALOADER_CONFIG_min_max_encoder_length_ratio': min_max_encoder_length_ratio,
        'DATALOADER_CONFIG_min_prediction_length': min_prediction_length,
        'DATALOADER_CONFIG_target_normalizer_transformation': target_normalizer_transformation,
        'DATALOADER_CONFIG_num_workers': num_workers,
        'DATALOADER_CONFIG_train_val_batch_size_ratio': train_val_batch_size_ratio,
    })
    
    ###start of dataloaders creation logic
    logger.info(f'Creating dataloaders')
    
    data = pd.read_pickle(preprocessed_data_input)
    training_cutoff = data['time_idx'].max() - max_prediction_length

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx = 'time_idx',
        target = 'volume',
        group_ids = ['agency', 'sku'],
        max_encoder_length = max_encoder_length,
        min_encoder_length = max_encoder_length // min_max_encoder_length_ratio,
        min_prediction_length = min_prediction_length,
        max_prediction_length = max_prediction_length,
        static_categoricals = ['agency', 'sku'],
        static_reals = ['avg_population_2017', 'avg_yearly_household_income_2017'],
        time_varying_known_categoricals = ['special_days', 'month'],
        variable_groups = {'special_days': special_days},
        time_varying_known_reals = ['time_idx', 'price_regular', 'discount_in_percent'],
        time_varying_unknown_categoricals = [],
        time_varying_unknown_reals = [
            'volume',
            'log_volume',
            'industry_volume',
            'soda_volume',
            'avg_max_temp',
            'avg_volume_by_agency',
            'avg_volume_by_sku',
        ],
        target_normalizer = GroupNormalizer(
            groups = ['agency', 'sku'], transformation = target_normalizer_transformation
        ),
        add_relative_time_idx = True,
        add_target_scales = True,
        add_encoder_length = True,
    )
    
    validation = TimeSeriesDataSet.from_dataset(training, data, predict = True, stop_randomization = True)
    
    train_dataloader = training.to_dataloader(train = True, batch_size = batch_size, num_workers = num_workers)
    val_dataloader = validation.to_dataloader(train = False, batch_size = batch_size * train_val_batch_size_ratio, num_workers = num_workers)
    ###end of dataloaders creation logic
    
    #saving data to component outputs    
    step_utils.save_data_to_component_output_in_pickle(training, training_output)
    step_utils.save_data_to_component_output_in_pickle(train_dataloader, train_loader_output)
    step_utils.save_data_to_component_output_in_pickle(val_dataloader, val_loader_output)
    
    #saving to GCS
    #saving training dataset
    training_dataset_gcs_path = f'{vertex_run_name}/dataloader_artifacts/training_dataset.pkl'   
    step_utils.upload_file_to_gcs(project, artifact_bucket, training_output, training_dataset_gcs_path)
    
    #saving train dataloader
    train_loader_gcs_path = f'{vertex_run_name}/dataloader_artifacts/train_dataloader.pkl' 
    step_utils.upload_file_to_gcs(project, artifact_bucket, train_loader_output, train_loader_gcs_path)

    #saving val dataloader
    val_loader_gcs_path = f'{vertex_run_name}/dataloader_artifacts/val_dataloader.pkl' 
    step_utils.upload_file_to_gcs(project, artifact_bucket, val_loader_output, val_loader_gcs_path)

    run.log_parms({
        'DATALOADER_OUTPUT_artifacts_directory_uri': f'gs://{artifact_bucket}/{vertex_run_name}/dataloader_artifacts/'
    })
    
    logger.info(f'Dataloader artifacts created and saved to GCS: gs://{artifact_bucket}/{vertex_run_name}/dataloader_artifacts/')