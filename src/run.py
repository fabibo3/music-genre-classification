# @author: Fabian Bongratz
#
# The main running procedure of the project
#
from argparse import ArgumentParser
from utils.experiments import run_experiment

def run():
    parser = ArgumentParser(description="Parameters")
    parser.add_argument("config_file",
                        nargs='+',
                        type=str,
                        help="The configuration used for the execution")
    print('Reading configuration files...')
    arguments = parser.parse_args()
    config_file_names = arguments.config_file
    # Run experiment for every config file
    for i, cf in enumerate(config_file_names):
        print(f"Start experiment {i+1} of {len(config_file_names)}")
        run_experiment(cf)



if __name__=='__main__':
    run()
