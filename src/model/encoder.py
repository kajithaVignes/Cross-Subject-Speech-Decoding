import numpy as np
import torch.nn as nn
import torch.nn.functional as F




class Encoder(nn.Module):
    """
    encoder
    """
    def __init__(self,input_dim,hidden_dim,phoneme_class):

        super().__init__()

        #tranformation affine 
        self.affine=nn.Linear(input_dim,input_dim)

        #early Gru Block which is bidirectionnel
        self.gru_Early =nn.GRU(input_dim,hidden_dim,num_layers=2,batch_first=True,bidirectional=True)
        self.phoneme_Early=nn.Linear(hidden_dim*2,phoneme_class)
        self.softmax_Early=nn.Softmax(dim=-1)
        self.projection_Early = nn.Linear(phoneme_class,hidden_dim)
        #il faudra faire une somme avant de passer au GRU du milieu h1 = z1 + p1
        #Midlle GRU Block which is bidirectionnel


        
        self.gru_Middle= nn.GRU(hidden_dim,hidden_dim,num_layers=2,batch_first=True,bidirectional=True)
        self.phoneme_Middle=nn.Linear(hidden_dim*2,phoneme_class)
        self.softmax_Middle=nn.Softmax(dim=-1)
        self.projection_Middle=nn.Linear(phoneme_class,hidden_dim)
        #Demême, que le early block, ne pas oublier de faire la somme h2 = z2 + ˆp2
        
        #Last GRU Block which is bidirectionnel
        self.gru_Final=nn.GRU(hidden_dim,hidden_dim,batch_first=True)
        nn.phoneme_Final=nn.Linear(hidden_dim,phoneme_class)


    def forward(self,X):

        