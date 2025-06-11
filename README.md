# gcp-vertex-ai-mlops-blueprint

use the principle of least privilige when granting service account permissions
you can use kfp build (with kubeflow pipeline artifacts in artifact registry) or docker images

which is better: Downloading data from gcs to local temp folder, or streaming download?

use:
gcloud config set auth/impersonate_service_account [service account email id]

refactor code
get versions for all requirements
get iam roles list
rm ^2 commits
update data directory with all the necessary gcs files.
add if else logic in model uploading