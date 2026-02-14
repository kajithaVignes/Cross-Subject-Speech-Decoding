import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):

    def __init__(self, input_dim: int, d: int, phoneme_class: int, blank_id: int = 0,learnable=True):
        super().__init__()
        assert d % 2 == 0

        self.input_dim = input_dim
        self.d = d
        self.phoneme_class = phoneme_class
        self.blank_id = blank_id
        self.learnable = learnable

        self.in_256_to_input = nn.Linear(256, input_dim) if input_dim != 256 else None

        self.linear = SessionAlignement(input_dim,learnable=self.learnable)

        bi_h = d // 2
        self.gru_early = nn.GRU(input_dim, bi_h, num_layers=2, batch_first=True, bidirectional=True)
        self.head_early = nn.Linear(d, phoneme_class)
        self.proj_early = nn.Linear(phoneme_class, d)

        self.gru_middle = nn.GRU(d, bi_h, num_layers=2, batch_first=True, bidirectional=True)
        self.head_middle = nn.Linear(d, phoneme_class)
        self.proj_middle = nn.Linear(phoneme_class, d)

        self.gru_final = nn.GRU(d, d, num_layers=1, batch_first=True, bidirectional=False)
        self.head_final = nn.Linear(d, phoneme_class)

    def _run_gru(self, gru, x, lens):
        x_pack = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        z_pack, _ = gru(x_pack)
        z, _ = pad_packed_sequence(z_pack, batch_first=True)
        return z

    def forward(self, X, input_len, session_key):

        if X.size(-1) != self.input_dim:
            if X.size(-1) == 256 and self.in_256_to_input is not None:
                X = self.in_256_to_input(X)
            else:
                raise RuntimeError(
                    f"Encoder expected feature dim {self.input_dim}, got {X.size(-1)}"
                )

        X = self.linear(X, session_key) 

        z1 = self._run_gru(self.gru_early, X, input_len) # (B,T,d)
        l1 = self.head_early(z1) # (B,T,N)
        p1 = self.proj_early(F.softmax(l1, dim=-1)) # (B,T,d)
        h1 = z1 + p1                                           

        z2 = self._run_gru(self.gru_middle, h1, input_len) # (B,T,d)
        l2 = self.head_middle(z2) # (B,T,N)
        p2 = self.proj_middle(F.softmax(l2, dim=-1)) # (B,T,d)
        h2 = z2 + p2                                             

        z3 = self._run_gru(self.gru_final, h2, input_len) # (B,T,d)
        l3 = self.head_final(z3) # (B,T,N)

        return l1, l2, l3, input_len



class SessionAlignement(nn.Module):
    def __init__(self, input_dim: int,learnable=True):
        super().__init__()
        self.input_dim = input_dim
        self.learnable = learnable
        self.transforms = nn.ModuleDict()

    def get(self, session_key, device=None) -> nn.Linear:
        if session_key not in self.transforms:
            
            if self.learnable:
                layer = nn.Linear(self.input_dim, self.input_dim, bias=True)
                nn.init.eye_(layer.weight)
                nn.init.zeros_(layer.bias)
        
            else:
                layer = nn.Identity()
                
            if device is not None:
                layer = layer.to(device) 
            self.transforms[session_key] = layer
        else:
            if device is not None:
                self.transforms[session_key] = self.transforms[session_key].to(device)
        return self.transforms[session_key]

    def forward(self, x: torch.Tensor, session_key):
        return self.get(session_key, device=x.device)(x)