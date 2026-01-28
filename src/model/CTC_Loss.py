import torch
import torch.nn as nn
import torch.nn.functionnal as F


class hierarchical_Loss(nn.Module):
    def __init__(self,balance):
        super().__init__()
        assert(0 < balance < 1)
        self.balance=balance
        self.CTC1=nn.CTCLoss()
        self.CTC2=nn.CTCLoss()
        self.CTC3=nn.CTCLoss()


    def forward(self,target,l1,l2,l3,input_len,target_len):
        "target doit être 1D "
        loss_early = self.CTC1(l1.transpose(0,1),target,input_len,target_len)
        loss_middle = self.CTC2(l2.transpose(0,1),target,input_len,target_len)
        loss_final = self.CTC3(l3.transpose(0,1),target,input_len,target_len)
        return loss_final + self.balance*(loss_early+ loss_middle)
        