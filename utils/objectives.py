import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCE2d(nn.Module):
    def __init__(self):
        super(WeightedBCE2d, self).__init__()

    def forward(self, input, target, negative_pixels):
        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        negative_pixels_t = negative_pixels.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)

        pos_index = (target_t > 0)
        hard_negative_index = (negative_pixels_t > 0)
        easy_negative_index = (negative_pixels_t == 0)

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        easy_negative_index = easy_negative_index.data.cpu().numpy().astype(bool)
        hard_negative_index = hard_negative_index.data.cpu().numpy().astype(bool)
        hard_negative_index_ = hard_negative_index.nonzero()

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()

        weight[pos_index] = 1.0
        weight[easy_negative_index] = 1.0
        weight[hard_negative_index] = 1.0 + negative_pixels_t[hard_negative_index_].cpu().numpy()

        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        loss = F.binary_cross_entropy(log_p, target_t, weight, size_average=True)

        return loss
