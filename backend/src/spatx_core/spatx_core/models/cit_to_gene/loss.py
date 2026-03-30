import torch
import torch.nn as nn
import torch.nn.functional as F

class SpearmanLoss(nn.Module):
    def __init__(self, reg=1.0): super().__init__(); self.reg=reg
    def forward(self, pred, true):
        # pred,true: [B,35]
        pr = self._soft_rank(pred); tr = self._soft_rank(true)
        pr = F.normalize(pr,dim=1); tr = F.normalize(tr,dim=1)
        corr = (pr*tr).sum(dim=1)
        return 1 - corr.mean()
    def _soft_rank(self, x):
        B,G = x.shape
        x2 = x.unsqueeze(-1)                     # [B,G,1]
        diff = x2 - x2.transpose(-1,-2)           # [B,G,G]
        P    = torch.sigmoid(-self.reg*diff)      # [B,G,G]
        return P.sum(dim=-1)   
    
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, reg=1.0):
        super().__init__()
        self.l1    = nn.L1Loss()
        self.sp    = SpearmanLoss(reg)
        self.alpha = alpha
    def forward(self, p,t):
        return self.l1(p,t) + self.alpha*self.sp(p,t)