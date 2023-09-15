import os, sys, yaml
from MegaViz.operation import perform

## FUNCTIONS ##
# method for loadining yaml files
def load_yaml(yaml_file_path):
    """ Load a yaml file and return a corresponding dictionary. """
    with open(yaml_file_path, 'r') as stream:
        yaml_dict = yaml.safe_load(stream)
    return yaml_dict

## ARGUMENTS ##
# path to YAML file containing instructions
config = sys.argv[1]

## SCRIPT ##
# load configurations from YAML
config_dict = load_yaml(config)

# perform visualization as described in configuration dictionary
for op in config_dict['operation']:
	perform(op, config_dict['operation'][op], config_dict['data'], config_dict['settings'], config_dict['model'])