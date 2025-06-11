from google.cloud import storage

def download_file_from_gcs(project, gcs_bucket, gcs_path, local_path):
    client = storage.Client(project=project)
    bucket = client.bucket(gcs_bucket)  
    blob = bucket.blob(gcs_path)  
    if blob.exists():  
        blob.download_to_filename(local_path) 
        return True
    else:
        return False

def upload_file_to_gcs(project, bucket_name, source_file_path, destination_blob_name):
    client = storage.Client(project=project)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_path)
