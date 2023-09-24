import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=[1, 1, 1], crossentropy=False):
        super(DiceLoss, self).__init__()
        self.eps = 1e-7
        self.crossentropy = crossentropy
        if crossentropy:
            self.ce = nn.CrossEntropyLoss()
        
        self.weight = weight

    def forward(self, y_pred, y_true):
        # y_pred = F.softmax(y_pred, dim=1)
        y_true = F.one_hot(y_true, num_classes=y_pred.size(1)).permute(0, 3, 1, 2).float()

        intersection = torch.sum(y_pred * y_true, dim=(0, 2, 3)) + self.eps
        union = torch.sum(y_pred, dim=(0, 2, 3)) + torch.sum(y_true, dim=(0, 2, 3)) + self.eps

        dice_coefficients = (2.0 * intersection) / union
        dice_loss = 1. - dice_coefficients.mean()
        dice_loss /= y_pred.size(1)

        if self.crossentropy:
            crossentropy_loss = self.ce(y_pred, y_true)
            total_loss = crossentropy_loss + dice_loss

            return total_loss

        else:
            return dice_loss