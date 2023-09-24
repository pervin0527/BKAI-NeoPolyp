import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassDiceScore(nn.Module):
    def __init__(self):
        super(MultiClassDiceScore, self).__init__()
        self.eps = 1e-7

    def forward(self, y_pred, y_true):
        y_true = F.one_hot(y_true, num_classes=y_pred.size(1)).permute(0, 3, 1, 2).float()

        intersection = torch.sum(y_pred * y_true, dim=(0, 2, 3)) + self.eps
        union = torch.sum(y_pred, dim=(0, 2, 3)) + torch.sum(y_true, dim=(0, 2, 3)) + self.eps

        dice_coefficients = (2.0 * intersection) / union
        mean_dice_score = dice_coefficients.mean()

        return mean_dice_score
