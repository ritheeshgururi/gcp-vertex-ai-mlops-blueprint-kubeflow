import json
import logging
import pandas as pd
from google.cloud import aiplatform
from utils.data_utils import upload_file_to_gcs, download_file_from_gcs

def batch_inference_step(
    project,
    location,
    gcs_bucket,
    data_path,
    endpoint_id,
    predictions_output
):
    logger = logging.getLogger(__name__)
    logger.info(f'Creating Batch Prediction')
    
    aiplatform.init(project=project, location=location)
    endpoint = aiplatform.Endpoint(endpoint_id)
    
    #download and process data
    local_path = '/tmp/raw_data.csv'
    download_file_from_gcs(project, gcs_bucket, data_path, local_path)
    
    data = pd.read_csv(local_path)
    data_list = data.values.tolist()
    
    json_input_data = json.dumps({'instances': data_list})
    
    predictions_endpoint = endpoint.raw_predict(body = json_input_data, headers = {'Content-Type': 'application/json'})
    predicted_data = pd.DataFrame(predictions_endpoint.json())
    
    predicted_data.to_csv('/tmp/model_predictions.csv', index=False)
    
    upload_file_to_gcs(project, gcs_bucket, '/tmp/model_predictions.csv', 'output/model_predictions.csv')

    logger.info('Batch Prediction completed successfully')