
import torch
import torch.nn as nn
import torch.nn.functional as F

class MCRMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def rmse(self, y_actual, y_pred):
        mse = self.mse(y_actual, y_pred)
        return torch.sqrt(mse)
    
    def forward(self, y_actual, y_pred, num_scored=None):
        if num_scored == None:
            num_scored = y_actual.shape[-1]
        score = 0
        for i in range(num_scored):
            score += self.rmse(y_actual[:, :, i], y_pred[:, :, i]) / num_scored
        return score
