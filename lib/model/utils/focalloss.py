import torch
import torch.nn as nn
import torch.nn.functional as F

class myFocalLoss(nn.Module):
    def __init__(self,weight=None,gama=5,reduction='mean'):
        super(myFocalLoss,self).__init__()
        self.gama = gama
        self.weight = weight
        self.reduction = reduction

    def forward(self,x,y):
        if len(list(x.size())) > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, len(list(x.size()))), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)
        ce_loss = F.cross_entropy(x,y)
        pt = torch.exp(-ce_loss)
        #print(ce_loss)
        focal_loss = ((1-pt) ** self.gama * ce_loss).mean()
        return focal_loss

if __name__ == '__main__':
    x = torch.randn(1, 3,3,4)
    y = torch.ones_like(x[:,0,:,:]).long()
    cri = myFocalLoss()
    focal_loss = cri(x,y)
    print(focal_loss)
