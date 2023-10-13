import pdb
import torch.nn as nn
import torch.nn.functional as F



class SemanticsLoss(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.name = 'semantics'
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, semanticMask1, logits, **kw):
        loss = 0
        # print(len(semanticMask1))
        # print(len(logits))
        for i in range(len(semanticMask1)):
            mask = semanticMask1[i]
            # print("before calculating loss:mask ", mask.shape)
            # print("before calculating loss:logits ", logits[0].shape)
            x = logits[0][i:i+1, :, :, :].squeeze(0).squeeze(0)
            # print("before calculating loss:logits ", x.shape)
            loss += self.loss_func(x, mask)
        return loss
