# music_foundation_model.py
#
# author: Jason Dominguez
# date: 2022-05-18
#
# source code for the music generation
# model implementation


# imports
from torch import nn


# class definitions
class FoundationModel(nn.Module):
    def __init__(self, ec_squared_vae_model, poly_chord_model):
        super(FoundationModel, self).__init__()