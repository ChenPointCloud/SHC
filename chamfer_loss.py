import torch
import torch.nn as nn


class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, x, y):
        P = self.batch_pairwise_dist(y, x)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.mean(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.mean(mins)

        return loss_1 + loss_2

    def batch_pairwise_dist(self, x, y):
        x = x.float()
        y = y.float()
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))

        dtype = torch.cuda.LongTensor

        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        # brk()
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P