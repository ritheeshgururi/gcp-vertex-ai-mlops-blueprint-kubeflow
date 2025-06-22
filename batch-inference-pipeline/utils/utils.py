import argparse

#get base image URL from CLI argument 
def get_base_image_path():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type = str, help = "The artifact registry URI to be used for building the base image")
    args = parser.parse_args()
    BASE_IMAGE_PATH = args.image_path
    return BASE_IMAGE_PATH