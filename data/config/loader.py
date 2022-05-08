# Copyright 2021 Lucas Fidon and Suprosanna Shit

import os
import yaml


def load_config(config_file=None):
    if config_file is None:  # default config file is nnUNet
        config_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "partialseg_fetal_challenge.yml"
        )
    else:
        if not os.path.isfile(config_file):
            raise FileNotFoundError('Config file %s not found' % config_file)
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
