import json
import time
import pandas as pd
from google.cloud import aiplatform

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
    print('Streaming prediction data from GCS')
    data = pd.read_csv(f'gs://{gcs_bucket}/{data_path}')
    data_list = data.values.tolist()
    
    json_input_data = json.dumps({'instances': data_list})
    
    print('Submitting prediction request to endpoint')
    start_time = time.time()
    predictions_endpoint = endpoint.raw_predict(body = json_input_data, headers = {'Content-Type': 'application/json'})
    end_time = time.time()
    print('Predictions received from endpoint')

    predicted_data = pd.DataFrame(predictions_endpoint.json())

    print(f'Streaming predictions to GCS path: gs://{gcs_bucket}/output/model_predictions.csv')
    predicted_data.to_csv(f'gs://{gcs_bucket}/output/model_predictions.csv', index=False)

    print(f'Endpoint prediction completed successfully in {end_time - start_time:.2f} seconds')

def main():
    PROJECT_ID = 'gcp-vertexai-mlops-blueprint'
    LOCATION = 'asia-south1'
    GCS_BUCKET = 'online-prediction-volume-forecasting'
    DATA_PATH = 'input/endpoint_prediction_dataset.csv'
    ENDPOINT_ID = '3758865217505722368'

    endpoint_prediction(
        project = PROJECT_ID,
        location = LOCATION,
        gcs_bucket = GCS_BUCKET,
        data_path = DATA_PATH,
        endpoint_id = ENDPOINT_ID
    )

if __name__ == '__main__':
    main()