import torch
import torch.nn as nn

class LogMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = 1e-6
        
    def forward(self, pred, actual):
        # adding 1 because log(1) = 0
        # pred += 1
        # actual += 1

        # # clipping to avoid log(0)
        # pred = torch.clamp(pred, min=self.eps)
        # actual = torch.clamp(actual, min=self.eps)

        # return nn.MSELoss()(torch.log(pred), torch.log(actual))
        # return nn.L1Loss()(torch.log(pred), torch.log(actual))

        return nn.MSELoss()(pred, actual)