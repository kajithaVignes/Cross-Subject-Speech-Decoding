import torch
import torch.nn as nn
import torch.nn.functional as F


class Hierarchical_Loss(nn.Module):
    def __init__(self,balance):
        super().__init__()
        assert(0 < balance < 1)
        self.balance=balance
        blank=0
        self.ctc1 = nn.CTCLoss(blank=blank, zero_infinity=True)
        self.ctc2 = nn.CTCLoss(blank=blank, zero_infinity=True)
        self.ctc3 = nn.CTCLoss(blank=blank, zero_infinity=True)


    def forward(self,target,l1,l2,l3,input_len,target_len):
        "target doit être 1D "

        lp1 = F.log_softmax(l1, dim=-1).transpose(0, 1)
        lp2 = F.log_softmax(l2, dim=-1).transpose(0, 1)
        lp3 = F.log_softmax(l3, dim=-1).transpose(0, 1)

        loss_early = self.ctc1(lp1,target,input_len,target_len)
        loss_middle = self.ctc2(lp2,target,input_len,target_len)
        loss_final = self.ctc3(lp3,target,input_len,target_len)
        return loss_final + self.balance*(loss_early+ loss_middle)
        