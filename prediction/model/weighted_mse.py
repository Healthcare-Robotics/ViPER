import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, force_weight=1.0, torque_weight=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.force_weight = force_weight
        self.torque_weight = torque_weight
        
    def forward(self, pred, actual):
        # print('pred.shape, gt.shape = ', pred.shape, actual.shape)
        force_pred = pred[:, :3]
        force_actual = actual[:, :3]
        torque_pred = pred[:, 3:]
        torque_actual = actual[:, 3:]

        loss = self.force_weight * self.mse(force_pred, force_actual) + self.torque_weight * self.mse(torque_pred, torque_actual)
        
        # print('weighted mse loss = ', loss)
        return loss