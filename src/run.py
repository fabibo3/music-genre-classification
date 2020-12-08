# @author: Fabian Bongratz
#
# The main running procedure of the project
#
import os
from argparse import ArgumentParser
from utils.experiments import run_experiment
from utils.folders import set_dataset_base_folder, get_dataset_base_folder

def run():
    parser = ArgumentParser(description="Parameters")
    parser.add_argument("--data_base_dir",
                        nargs='?',
                        dest='data_base_dir',
                        action='store_const',
                        type=str,
                        default=None,
                        help="The base directory of all data")
    parser.add_argument("config_file",
                        nargs='+',
                        type=str,
                        help="The configuration used for the execution")
    print('Reading configuration files...')
    arguments = parser.parse_args()
    config_file_names = arguments.config_file
    data_base_dir = arguments.data_base_dir

    if(not os.path.isdir(get_dataset_base_folder())):
        if(data_base_dir is not None):
            set_data_base_folder(data_base_dir)
        else:
            raise ValueError("Please enter data directory")

    # Run experiment for every config file
    for i, cf in enumerate(config_file_names):
        print(f"Start experiment {i+1} of {len(config_file_names)}")
        run_experiment(cf)



if __name__=='__main__':
    run()
