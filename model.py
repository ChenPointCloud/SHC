#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import global_spacial_consistency


# Part of the code is referred from: http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
# Part of the code is referred from: https://github.com/WangYueFt/dcp

def nearest_neighbor(src, dst):
    inner = -2 * torch.matmul(src.transpose(1, 0).contiguous(), dst)  # src, dst (num_dims, num_points)
    distances = -torch.sum(src ** 2, dim=0, keepdim=True).transpose(1, 0).contiguous() - inner - torch.sum(dst ** 2,
                                                                                                           dim=0,
                                                                                                           keepdim=True)
    distances, indices = distances.topk(k=1, dim=-1)
    return distances, indices

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20):
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)

    return feature

class DGCNN(nn.Module):
    def __init__(self, emb_dims=512):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        return x

class Corresponodences(nn.Module):
    def __init__(self, args):
        super(Corresponodences, self).__init__()
        self.emb_nn = DGCNN(emb_dims=args.emb_dims)

    def forward(self, src, tgt, return_feature=False):
        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding)

        scores = torch.softmax(scores, dim=2)

        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())

        if return_feature:
            return src_corr, src_embedding, tgt_embedding, scores
        else:
            return src_corr

class SHCNet(nn.Module):
    def __init__(self, args):
        super(SHCNet, self).__init__()
        self.emb_nn = DGCNN(emb_dims=args.emb_dims)

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding)

        _, cor_index = torch.max(scores, dim=2)
        src_corr = tgt[:, :, cor_index[0]]

        return src_corr

def Registration(src, src_corr, consistency=False, auto_threshold=False):
    if consistency:
        src, src_corr = global_spacial_consistency(src, src_corr, min_len=200, auto_threshold=auto_threshold)
        src, src_corr = global_spacial_consistency(src, src_corr, min_len=70, auto_threshold=auto_threshold)
        src, src_corr = global_spacial_consistency(src, src_corr, min_len=20, auto_threshold=auto_threshold)

    src_centered = src - src.mean(dim=2, keepdim=True)
    src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

    H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())

    R = []

    for i in range(src.size(0)):
        u, s, v = torch.svd(H[i])
        r = torch.matmul(v, u.transpose(1, 0).contiguous())
        r_det = torch.det(r)
        if r_det < 0:
            u, s, v = torch.svd(H[i])
            reflect = torch.eye(3)
            reflect[2, 2] = -1
            reflect = reflect.cuda()
            v = torch.matmul(v, reflect)
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
        R.append(r)

    R = torch.stack(R, dim=0)
    t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)

    return R, t.view(src.size(0), 3)


