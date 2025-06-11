from google.cloud import storage, aiplatform
import pickle

def download_file_from_gcs(project, gcs_bucket, gcs_path, local_path):
    """Donloads a file from a GCS bucket"""
    
    client = storage.Client(project=project)
    bucket = client.bucket(gcs_bucket)
    blob = bucket.blob(gcs_path)
    if blob.exists():
        blob.download_to_filename(local_path)
        return True
    else:
        return False

def upload_file_to_gcs(project, bucket_name, source_file_path, destination_blob_name):
    """Uploads a file to the specified GCS bucket"""

    client = storage.Client(project=project)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_path)

    print(f'File {source_file_path} uploaded to {bucket_name}/{destination_blob_name}.')
    
def load_data_from_component_input_in_pickle(component_input):
    with open(component_input, 'rb') as f:
        target_data = pickle.load(f)
    return target_data
    
def save_data_to_component_output_in_pickle(source_data, component_output):
    with open(component_output, 'wb') as f:
        pickle.dump(source_data, f)
        
def get_experiment_run(project, location, experiment_name, run_name):
    """Initializes Vertex AI and returns an experiment run object"""

    aiplatform.init(
        project = project,
        location = location,
        experiment = experiment_name
    )

    run = aiplatform.ExperimentRun(
        run_name = run_name,
        experiment = experiment_name
    )
    return run