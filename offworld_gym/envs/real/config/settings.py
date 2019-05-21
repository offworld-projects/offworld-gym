import yaml
import os 
config = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "settings.yaml")))