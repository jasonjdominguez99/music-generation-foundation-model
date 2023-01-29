# music_foundation_model.py
#
# author: Jason Dominguez
# date: 2022-05-18
#
# source code for the music generation
# model implementation


# imports
from torch import nn
from transformer import VanillaTransformerModel


# class definitions
class MusicFoundationModel(nn.Module):
    def __init__(self, melody_model, harmony_model, ntokens,
                 emsize, d_hid, nlayers, nhead, dropout):
        super().__init__()
        self.melody_model = melody_model
        self.harmony_model = harmony_model
        
        # TODO: define the transformer which uses the pre-trained models
        self.transformer = VanillaTransformerModel(
            ntokens, emsize, nhead, d_hid, nlayers, dropout
        )

    # TODO: implement training method    
    # TODO: implement inference method
    def forward(self, melody, pr_mat, condition):
        z_chd, z_txt = self.harmony_model()
