import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
        self.projection_Early = nn.Linear(phoneme_class,hidden_dim*2)
        #il faudra faire une somme avant de passer au GRU du milieu h1 = z1 + p1
        #Midlle GRU Block which is bidirectionnel

        
        self.gru_Middle= nn.GRU(hidden_dim*2,hidden_dim,num_layers=2,batch_first=True,bidirectional=True)
        self.phoneme_Middle=nn.Linear(hidden_dim*2,phoneme_class)
        self.softmax_Middle=nn.Softmax(dim=-1)
        self.projection_Middle=nn.Linear(phoneme_class,hidden_dim*2)
        #Demême, que le early block, ne pas oublier de faire la somme h2 = z2 + ˆp2
        
        #Last GRU Block which is bidirectionnel
        self.gru_Final=nn.GRU(hidden_dim*2,hidden_dim,batch_first=True)
        self.phoneme_Final=nn.Linear(hidden_dim,phoneme_class)


    def forward(self, X, input_len):

        X = self.affine(X)

        def run_gru(gru, x, lens): # Avoid learning from padded parts
            x_pack = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
            z_pack, _ = gru(x_pack)
            z, _ = pad_packed_sequence(z_pack, batch_first=True)
            return z

        # Early GRU
        z1 = run_gru(self.gru_Early, X, input_len)
        l1 = self.phoneme_Early(z1)

        p1 = self.projection_Early(F.softmax(l1, dim=-1))
        h1 = z1 + p1

        # Middle GRU
        z2 = run_gru(self.gru_Middle, h1, input_len)
        l2 = self.phoneme_Middle(z2)

        p2 = self.projection_Middle(F.softmax(l2, dim=-1))
        h2 = z2 + p2

        # Final GRU
        z3 = run_gru(self.gru_Final, h2, input_len)
        l3 = self.phoneme_Final(z3)

        return l1, l2, l3, input_len


        