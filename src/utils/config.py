# @author: Fabian Bongratz
#
# Utility functions for configuration reading and/or processing

import json
from utils.folders import get_config_file_path

# Load configuration file (.json) into a dict
def load_config_file(file_name: str):
    full_path = get_config_file_path(file_name)
    with open(full_path) as json_file:
        config = json.load(json_file)
    return config

