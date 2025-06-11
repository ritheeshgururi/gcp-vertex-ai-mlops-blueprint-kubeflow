# gcp-vertex-ai-mlops-blueprint

use the principle of least privilige when granting service account permissions
you can use kfp build (with kubeflow pipeline artifacts in artifact registry) or docker images

which is better: Downloading data from gcs to local temp folder, or streaming download?

use:
gcloud config set auth/impersonate_service_account [service account email id]

modify streaming and gcs upload logic
refactor code