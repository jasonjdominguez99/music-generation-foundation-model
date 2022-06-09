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

from ec_squared_vae.code.ec_squared_vae import ECSquaredVAE
from polyphonic_chord_texture_disentanglement.train import (
    define_poly_chord_model
)

from music_foundation_model import MusicFoundationModel


def load_ec_squared_vae(config_file_path):
    with open(config_file_path) as f:
        args = json.load(f)
    
    load_path = "ec-squared-vae/params/{}.pt".format(args["name"])

    model = ECSquaredVAE(
        args["roll_dim"], args["hidden_dim"], args["rhythm_dim"], 
        args["condition_dim"], args["pitch_dim"],
        args["rhythm_dim"], args["time_step"]
    )
    model.load_state_dict(torch.load(load_path))

    return model


def load_poly_chord_model(config_file_path):
    with open(config_file_path) as f:
        args = json.load(f)

    # TODO: update load path with the correct DATE and TIME%H%M%S
    load_path = "/polyphonic_chord_texture_disentanglement\
                 /result_DATE_TIME%H%M%S/models\
                 /{}_final.pt".format(args["name"])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = define_poly_chord_model(args, device)
    model.load_state_dict(torch.load(load_path))
    
    return model


def load_pre_trained_models(config_file_paths):
    ec_squared_vae_model = load_ec_squared_vae(
        config_file_paths["ec_squared_vae"]
    )
    poly_chord_model = load_poly_chord_model(
        config_file_paths["poly_chord"]
    )

    return ec_squared_vae_model, poly_chord_model


def train_foundation_model(model):
    # TODO: implement the logic for training the foundation model
    pass


def main():
    foundation_model_config_file_path = "music_foundation_model_config.json"
    pre_trained_models_config_paths = {
        "ec_squared_vae": "ec_squared_vae/code\
                           /ec_squared_vae_model_config.json",
        "poly_chord": "polyphonic_chord_texture_disentanglement\
                       /poly_chord_model_config.json"
    }

    melody_model, harmony_model = load_pre_trained_models(
        pre_trained_models_config_paths
    )

    fdn_model = MusicFoundationModel(
        melody_model, harmony_model
    )

if __name__ == "__main__":
    main()