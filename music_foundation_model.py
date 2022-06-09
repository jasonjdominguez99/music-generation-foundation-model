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
class MusicFoundationModel(nn.Module):
    def __init__(self, melody_model, harmony_model):
        super(MusicFoundationModel, self).__init__()
        self.melody_model = melody_model
        self.harmony_model = harmony_model
        
        # TODO: define the transformer which uses the pre-trained models
        # TODO: implement training method
        # TODO: implement inference method
