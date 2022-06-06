# main.py
#
# author: Jason Dominguez
# date: 2022-05-18
#
# main source code for implementing the
# music generation foundation model


# imports
import json
import torch

from ec_squared_vae.code.ec_squared_vae import ECSquaredVae


def load_ec_squared_vae(load_path, config_file_path):
    with open(config_file_path) as f:
        args = json.load(f)
    
    load_path = "ec-squared-vae/params/{}".format(args["name"])

    model = 
    model.load_state_dict(torch.load(load_path))


def load_pre_trained_models(config_file_paths):
    ec_squared_vae_model = load_ec_squared_vae(
        config_file_paths["ec_squared_vae"]
    )


def main():
    foundation_model_config_file_path = "music_foundation_model_config.json"
    pre_trained_model_config_paths = {
        "ec_squared_vae": "ec-squared-vae/code/ec_squared_vae_model_config.json",
        "poly_chord": ""
    }

    load_pre_trained_models(pre_trained_model_config_paths)

if __name__ == "__main__":
    main()