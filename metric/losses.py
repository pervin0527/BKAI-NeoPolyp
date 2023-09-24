import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyLoss(nn.Module):
    def __init__(self, num_classes=3, weight=[1, 1, 1], alpha=0.5, beta=0.5, crossentropy=False):
        super(TverskyLoss, self).__init__()
        self.eps = 1e-7
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.crossentropy = crossentropy
        if crossentropy:
            self.ce = nn.CrossEntropyLoss()
        
        self.weight = weight

    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        y_true = F.one_hot(y_true, num_classes=y_pred.size(1)).permute(0, 3, 1, 2).float()

        intersection = torch.sum(y_pred * y_true, dim=(2, 3))
        false_pos = torch.sum(y_pred * (1 - y_true), dim=(2, 3))
        false_neg = torch.sum((1 - y_pred) * y_true, dim=(2, 3))

        tversky = (intersection + self.eps) / (intersection + self.alpha * false_pos + self.beta * false_neg + self.eps)
        tversky_loss = (1 - tversky).mean()
        tversky_loss /= self.num_classes

        if self.crossentropy:
            crossentropy_loss = self.ce(y_pred, y_true.argmax(dim=1))
            total_loss = crossentropy_loss + tversky_loss
            return total_loss

        else:
            return tversky_loss



class DiceLoss(nn.Module):
    def __init__(self, num_classes=3, weight=[1, 1, 1], crossentropy=False):
        super(DiceLoss, self).__init__()
        self.eps = 1e-7
        self.num_classes = num_classes
        self.crossentropy = crossentropy
        if crossentropy:
            self.ce = nn.CrossEntropyLoss()
        
        self.weight = weight

    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        y_true = F.one_hot(y_true, num_classes=y_pred.size(1)).permute(0, 3, 1, 2).float()

        intersection = torch.sum(y_pred * y_true, dim=(2, 3))
        union = torch.sum(y_pred, dim=(2, 3)) + torch.sum(y_true, dim=(2, 3))

        dice_coeff = (2. * intersection + self.eps) / (union + self.eps)
        dice_loss = 1. - dice_coeff.mean()
        dice_loss /= self.num_classes

        if self.crossentropy:
            crossentropy_loss = self.ce(y_pred, y_true)
            # total_loss = 0.4 * crossentropy_loss + 0.6 * dice_loss
            total_loss = crossentropy_loss + dice_loss

            return total_loss

        else:
            return dice_loss
        

# class DiceLoss(nn.Module):
#     def __init__(self, num_classes=3, crossentropy=False, weight=None):
#         super(DiceLoss, self).__init__()
#         self.num_classes = num_classes
#         self.crossentropy = crossentropy
#         if crossentropy:
#             self.ce = nn.CrossEntropyLoss()
#         self.weight = weight

#     def one_hot_encoder(self, input_tensor):
#         tensor_list = []
#         for i in range(self.num_classes):
#             temp_prob = input_tensor == i
#             tensor_list.append(temp_prob.unsqueeze(1))
#         output_tensor = torch.cat(tensor_list, dim=1)

#         return output_tensor.float()

#     def dice_loss(self, score, target):
#         target = target.float()
#         smooth = 1e-5
#         intersect = torch.sum(score * target)
#         y_sum = torch.sum(target * target)
#         z_sum = torch.sum(score * score)
#         loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
#         loss = 1 - loss

#         return loss

#     def forward(self, y_pred, y_true):
#         if self.weight == None:
#             self.weight = [1] * self.num_classes

#         y_pred = torch.softmax(y_pred, dim=1)
#         y_true_one_hot = self.one_hot_encoder(y_true)

#         assert y_pred.size() == y_true_one_hot.size(), 'predict {} & y_true {} shape do not match'.format(y_pred.size(), y_true_one_hot.size())

#         loss = 0.0
#         for i in range(0, self.num_classes):
#             dice = self.dice_loss(y_pred[:, i, :, :], y_true_one_hot[:, i, :, :])
#             loss += dice * self.weight[i]

#         dice_loss = loss / self.num_classes

#         if not self.crossentropy:
#             return dice_loss
        
#         else:
#             cross_entropy_loss = self.ce(y_pred, y_true)
#             # final_loss = 0.4 * cross_entropy_loss + 0.6 * dice_loss
#             final_loss = cross_entropy_loss + dice_loss
#             return final_loss