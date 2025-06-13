import logging
import pandas as pd
import numpy as np

from configuration.step_config import PreprocessConfig
from utils import step_utils

def preprocess_step(
    project,
    location,
    artifact_bucket,
    data_bucket,
    data_path,
    preprocessed_data_path,
    vertex_experiment_name,
    vertex_run_name
):
    """Fetches and preprocesses the Volume Forecasting Data"""

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    ###fetching preprocessing variables/parameters from the step_config.py file
    special_days = PreprocessConfig.special_days
    volume_transform_type = PreprocessConfig.volume_transform_type
    
    ###logging parameters to the vertex experiment run
    logger.info('Initializing Vertex AI with experiment name and instantiating experiment run')
    run = step_utils.get_vertex_experiment_run(
        project,
        location,
        vertex_experiment_name,
        vertex_run_name
    )

    logger.info('Logging step config parameters to the experiment run')
    run.log_params({
        'PREPROCESS_CONFIG_special_days': str(special_days),
        'PREPROCESS_CONFIG_volume_transform_type': volume_transform_type
    })
    
    ###start of preprocessing logic
    logger.info('Starting preprocessing')
    
    #stream data directly from GCS with pandas using gcsfs
    logger.info(f'Streaming data from GCS path - gs://{data_bucket}/{data_path} using gcsfs')
    data = pd.read_csv(f'gs://{data_bucket}/{data_path}')
    
    data['date'] = pd.to_datetime(data['date'])
    
    #add time index
    data['time_idx'] = data['date'].dt.year * 12 + data['date'].dt.month
    data['time_idx'] -= data['time_idx'].min()
    
    #add features
    data['month'] = data.date.dt.month.astype(str).astype('category')
    data['log_volume'] = np.log(data.volume + 1e-8)
    data['avg_volume_by_sku'] = data.groupby(['time_idx', 'sku'], observed = True).volume.transform(volume_transform_type)
    data['avg_volume_by_agency'] = data.groupby(['time_idx', 'agency'], observed = True).volume.transform(volume_transform_type)
    
    data[special_days] = data[special_days].apply(lambda x: x.map({0: '-', 1: x.name})).astype('category')
    ###end of preprocessing logic
    
    #saving to component output
    data.to_pickle(preprocessed_data_path)
    
    #saving to GCS
    preprocessed_gcs_path = f'{vertex_run_name}/preprocessing_artifacts/preprocessed_data.pkl'
    logger.info(f'Archiving preprocessed data to GCS path: gs://{artifact_bucket}/{preprocessed_gcs_path}')
    step_utils.upload_file_to_gcs(
        project,
        artifact_bucket,
        preprocessed_data_path,
        preprocessed_gcs_path
    )
    logger.info(f'Preprocessed data archived to GCS path: gs://{artifact_bucket}/{preprocessed_gcs_path}')
    
    run.log_params({
        'PREPROCESS_OUTPUT_gcs_uri': f'gs://{artifact_bucket}/{preprocessed_gcs_path}'
    })
    logger.info('Preprocessing component completed and output artifacts logged to the experiment run')