#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function

import torch
import numpy as np
from scipy.spatial.transform import Rotation


# Part of the code is referred from: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py

def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def transform_point_cloud(point_cloud, rotation, translation):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_matrix(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')


def cloudNearTrans(x_in, y_in):
    x = x_in.clone().float()
    y = y_in.clone().float()
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    dtype = torch.cuda.LongTensor
    diag_ind_x = torch.arange(0, num_points_x).type(dtype)
    diag_ind_y = torch.arange(0, num_points_y).type(dtype)
    rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    minIndex = torch.argmin(P, dim=2)
    for i in range(x.size(0)):
        x[i] = y[i][minIndex[i]]
    return x

def spacial_consistency(src, target, k, threshold = 0.001, min_len = 50):#B, C, N
    src, target = src.transpose(1, 2), target.transpose(1, 2)
    n = src.size(1)
    batch = src.size(0)
    target_expend = torch.cat((target, target[:, :k, :]), dim=1)
    src_expend = torch.cat((src, src[:, :k, :]), dim=1)
    vote = torch.zeros(src.shape[:2]).cuda()

    current_s = threshold
    while True:
        for i in range(k):
            d_t = torch.sqrt(torch.sum(torch.pow(target-target_expend[:,i:i+n,:], 2), dim=2))
            d_s = torch.sqrt(torch.sum(torch.pow(src-src_expend[:,i:i+n,:], 2), dim=2))
            d = torch.abs(d_s - d_t)
            mask = torch.le(d, current_s)
            vote += mask
        len = torch.min(torch.sum(torch.ge(vote, k//2), dim=1))
        if len > min_len:
            break
        current_s += threshold


    # print('len:%f, threshold:%f'%(len, current_s))
    src_valid = []
    target_valid = []
    for i in range(batch):
        mask = torch.ge(vote[i], k//2)
        src_valid.append(src[i][mask][:len])
        target_valid.append(target[i][mask][:len])
    src_valid = torch.stack(src_valid)
    target_valid = torch.stack(target_valid)
    return src_valid.transpose(1, 2), target_valid.transpose(1, 2)

def pairwise_distance(x):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = xx + inner + xx.transpose(2, 1).contiguous()
    return pairwise_distance

def pairwise_distance_batch(x,y):
    xx = torch.sum(torch.mul(x,x), 1, keepdim = True)#[b,1,n]
    yy = torch.sum(torch.mul(y,y),1, keepdim = True) #[b,1,n]
    inner = -2*torch.matmul(x.transpose(2,1),y) #[b,n,n]
    pair_distance = xx.transpose(2, 1) + inner + yy #[b,n,n]
    pair_distances = torch.sqrt(pair_distance)
    return pair_distances #pij->xi-yj


def global_spacial_consistency(src, target, threshold=0.0001, min_len=50, auto_threshold=False):#B, C, N
    src_dis = pairwise_distance(src)
    tgt_dis = pairwise_distance(target)
    diff = torch.abs(src_dis - tgt_dis)
    current_s = threshold
    while True:
        mask = torch.le(diff, current_s)
        vote = torch.sum(mask, dim=2)
        value, _ = vote.topk(min_len)
        qualified = torch.min(value) * 0.5
        if qualified < 10:
            if auto_threshold:
                current_s *= 2
                continue
            qualified = 10
        mask = torch.ge(vote, qualified)
        src_valid = src[0].transpose(0, 1)[mask[0]]
        if src_valid.size(0) > min_len:
            break
        current_s *= 2

    target_valid = target[0].transpose(0, 1)[mask[0]]
    src_valid = src_valid.unsqueeze(0)
    target_valid = target_valid.unsqueeze(0)
    return src_valid.transpose(1, 2), target_valid.transpose(1, 2)


def correspondence(x, y):
    dis = pairwise_distance_batch(x, y)
    min_dis, index = torch.min(dis, dim=2)
    mask = torch.le(min_dis, 0.0001)
    return index, mask

def correspondence_loss(x, y, scores):
    num_points = x.size(2)
    batch_size = x.size(0)
    y_num_points = y.size(2)
    dis = pairwise_distance_batch(x, y)
    min_dis, index = torch.min(dis, dim=2)
    mask = torch.le(min_dis, 0.001)
    off = torch.arange(0, num_points, device='cuda').view(1, -1) * y_num_points
    index = index + off
    idx_base = torch.arange(0, batch_size, device='cuda').view(-1, 1) * num_points * y_num_points
    index = index + idx_base
    index = index.view(batch_size * num_points)
    scores = 1 - scores
    scores = scores.view(-1)
    min_feature = scores[index].view(batch_size, -1)
    loss = min_feature * mask
    loss = torch.sum(loss) / num_points
    return loss
