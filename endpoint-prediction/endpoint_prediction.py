import json
import time
import pandas as pd
from google.cloud import aiplatform
from utils.data_utils import upload_file_to_gcs, download_file_from_gcs

def endpoint_prediction(
    project,
    location,
    gcs_bucket,
    data_path,
    endpoint_id,
):
    aiplatform.init(project=project, location=location)

    endpoint = aiplatform.Endpoint(endpoint_id)
    print(f'Using endpoint: {endpoint.display_name} ({endpoint.name})')

    #download and process data
    print('Downloading prediction data from GCS')
    local_path = '/tmp/raw_data.csv'
    download_file_from_gcs(project, gcs_bucket, data_path, local_path)
    print('Data downloading complete')
    
    data = pd.read_csv(local_path)
    data_list = data.values.tolist()
    
    json_input_data = json.dumps({'instances': data_list})
    
    print('Submitting prediction request to endpoint')
    start_time = time.time()
    predictions_endpoint = endpoint.raw_predict(body = json_input_data, headers = {'Content-Type': 'application/json'})
    end_time = time.time()
    print('Predictions received from endpoint')

    predicted_data = pd.DataFrame(predictions_endpoint.json())
    predicted_data.to_csv('/tmp/model_predictions.csv', index=False)
    
    print(f'Uploading predictions to GCS path: gs://{gcs_bucket}/output/model_predictions.csv')
    upload_file_to_gcs(project, gcs_bucket, '/tmp/model_predictions.csv', 'output/model_predictions.csv')

    print(f'Endpoint prediction completed successfully in {end_time - start_time:.2f} seconds')

def main():
    PROJECT_ID = 'gcp-vertexai-mlops-blueprint'
    LOCATION = 'asia-south1'
    GCS_BUCKET = 'online-prediction-volume-forecasting'
    DATA_PATH = 'input/endpoint_prediction_dataset.csv'
    ENDPOINT_ID = '7615529795278864384'

    endpoint_prediction(
        project = PROJECT_ID,
        location = LOCATION,
        gcs_bucket = GCS_BUCKET,
        data_path = DATA_PATH,
        endpoint_id = ENDPOINT_ID
    )

if __name__ == '__main__':
    main()