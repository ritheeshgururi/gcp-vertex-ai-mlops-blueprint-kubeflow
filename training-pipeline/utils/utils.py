import argparse
import re

def get_base_image_path():
    """Takes the image_path and image_tag arguments for use in downstream pipeline"""
    
    parser = argparse.ArgumentParser()

    parser.add_argument('image_path', type = str, help = 'The artifact registry path to the base image that is to be used to build the pipeline components')
    
    args = parser.parse_args()

    BASE_IMAGE_PATH = args.image_path
    
    return BASE_IMAGE_PATH