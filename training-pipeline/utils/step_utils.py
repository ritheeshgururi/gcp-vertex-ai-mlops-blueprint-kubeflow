from google.cloud import storage
import pickle

def download_file_from_gcs(project, gcs_bucket, gcs_path, local_path):
    """Donloads a file from a GCS bucket"""
    
    client = storage.Client(project=project)
    bucket = client.bucket(gcs_bucket)  # Get the bucket object
    blob = bucket.blob(gcs_path)  # Get the blob object
    if blob.exists():  # Check if the blob exists
        blob.download_to_filename(local_path)  # Download the file
        return True
    else:
        return False

def upload_file_to_gcs(project, bucket_name, source_file_path, destination_blob_name):
    """Uploads a file to the specified GCS bucket"""
    # Initialize a GCS client
    from google.cloud import storage

    client = storage.Client(project=project)

    # Get the bucket
    bucket = client.get_bucket(bucket_name)

    # Create a blob in the bucket (essentially a reference to the file)
    blob = bucket.blob(destination_blob_name)

    # Upload the file to GCS
    blob.upload_from_filename(source_file_path)

    print(f'File {source_file_path} uploaded to {bucket_name}/{destination_blob_name}.')
    
def load_data_from_component_input_in_pickle(component_input):
    with open(component_input, 'rb') as f:
        target_data = pickle.load(f)
    return target_data
    
def save_data_to_component_output_in_pickle(source_data, component_output):
    with open(component_output, 'wb') as f:
        pickle.dump(source_data, f)