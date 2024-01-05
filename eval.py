from __future__ import print_function
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse


import torch
import torch.nn as nn
import torch.nn.functional as F

import util
from util import global_spacial_consistency


import model
from data import ModelNet40
from model import SHCNet, Corresponodences
from util import transform_point_cloud, npmat2euler,cloudNearTrans
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from chamfer_loss import ChamferLoss

from data_icl import TestData
import dataset

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()

chamfer = ChamferLoss()


def cor_loss_compute(src, target, generate, rotation_ab, translation_ab, points=False):
    st = transform_point_cloud(src, rotation_ab, translation_ab)
    src_corr, src_embedding, tgt_embedding, scores = generate(src, target, True)
    rotation_ab_pred, translation_ab_pred = model.Registration(src, src_corr)
    loss_corr = util.correspondence_loss(st, target, scores)
    loss = loss_corr
    if not points:
        return [loss, 0, 0, 0], rotation_ab_pred, translation_ab_pred
    else:
        dis = util.pairwise_distance_batch(st, target)
        min_dis, index_dis = torch.min(dis, dim=2)
        min_score, index_score = torch.max(scores, dim=2)
        diff = index_score -index_dis
        ones = torch.ones_like(diff)
        ret = torch.where(diff==0, diff, ones)
        ret = 1 - ret
        mask = torch.le(min_dis, 0.001)
        ret = ret * mask
        num = torch.sum(ret)
        return [loss, 0, 0, 0], rotation_ab_pred, translation_ab_pred, num



def shc_one_epoch(args, net, test_loader):
    net.eval()
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []

    for src, target, rotation_ab, translation_ab in tqdm(test_loader):
        src = src.cuda()
        target = target.cuda()
        rotation_ab = rotation_ab.cuda()
        translation_ab = translation_ab.cuda()

        src_corr = net(src, target)
        rotation_ab_pred, translation_ab_pred = model.Registration(src, src_corr, True)

        for i in range(2):
            src_i = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)
            src_corr = net(src_i, target)
            src_corr = cloudNearTrans(src_corr.transpose(1, 2), target.transpose(1, 2)).transpose(1, 2)
            if args.dataset == 'icl_nuim':
                rotation_ab_pred_i, translation_ab_pred_i = model.Registration(src_i, src_corr, True, True)
            else:
                rotation_ab_pred_i, translation_ab_pred_i = model.Registration(src_i, src_corr, True)
            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) \
                                  + translation_ab_pred_i

        st = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)
        src_corr = cloudNearTrans(st.transpose(1, 2), target.transpose(1, 2)).transpose(1, 2)
        if args.dataset == 'icl_nuim':
            src, src_corr = global_spacial_consistency(src, src_corr, auto_threshold=True)
        else:
            src, src_corr = global_spacial_consistency(src, src_corr, auto_threshold=True)
        rotation_ab_pred, translation_ab_pred = model.Registration(src, src_corr)


        ## save rotation and translation
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())

    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)


    return rotations_ab, translations_ab, rotations_ab_pred, translations_ab_pred


#no SHCNet frame
def base_one_epoch(args, net, test_loader, points=False):
    net.eval()

    total_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []

    global_losses = 0
    trans_losses = 0
    gene_losses = 0
    num_inlier = 0
    for src, target, rotation_ab, translation_ab in tqdm(test_loader):
        src = src.cuda()
        target = target.cuda()
        rotation_ab = rotation_ab.cuda()
        translation_ab = translation_ab.cuda()

        batch_size = src.size(0)
        num_examples += batch_size

        if points:
            [loss, loss_global, trans_loss, gene_loss], rotation_ab_pred, translation_ab_pred, num = cor_loss_compute(src, target, net, rotation_ab, translation_ab, points)
        else:
            [loss, loss_global, trans_loss, gene_loss], rotation_ab_pred, translation_ab_pred = cor_loss_compute(
                src, target, net, rotation_ab, translation_ab, points)

        if points:
            num_inlier += num
        total_loss += loss.item() * batch_size

        global_losses += loss_global * batch_size
        trans_losses += trans_loss * batch_size
        gene_losses += gene_loss * batch_size

        ## save rotation and translation
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())

    num_inlier = num_inlier * 1.0 / num_examples
    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)


    if points:
        return total_loss * 1.0 / num_examples, \
               rotations_ab, translations_ab, rotations_ab_pred, translations_ab_pred, num_inlier
    return total_loss * 1.0 / num_examples,\
           rotations_ab, translations_ab, rotations_ab_pred, translations_ab_pred

def get_args():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--use_sgd', action='store_true', default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--num_points', type=int, default=1024, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40', 'icl_nuim', '7scenes'], metavar='N',
                        help='dataset to use')
    parser.add_argument('--factor', type=float, default=4, metavar='N',
                        help='Divided factor for rotations')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--sub_points', type=int, default=1024, metavar='N',
                        help='partial overlapping')

    args = parser.parse_args()
    return args


def eval(args, epoch):
    net = SHCNet(args).cuda()
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.eval()
    if args.dataset == 'modelnet40':
        test_loader = DataLoader(
            ModelNet40(num_points=args.num_points, partition='test', gaussian_noise=args.gaussian_noise,
                       unseen=args.unseen, factor=args.factor, sub_points=args.sub_points),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    elif args.dataset == 'icl_nuim':
        test_data = TestData(args)
        test_loader = DataLoader(test_data, args.test_batch_size)
    elif args.dataset == '7scenes':
        testset = dataset.get_datasets(args)
        test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.test_batch_size, shuffle=False)
    else:
        raise Exception("not implemented")

    net.load_state_dict(torch.load('checkpoints/%s/models/model.%s.t7' % (args.exp_name, str(epoch))), strict=False)
    test_rotations_ab, test_translations_ab, \
    test_rotations_ab_pred, test_translations_ab_pred = shc_one_epoch(args, net, test_loader)

    test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
    test_r_mse_ab = np.mean((test_rotations_ab_pred_euler - npmat2euler(test_rotations_ab)) ** 2)
    test_r_rmse_ab = np.sqrt(test_r_mse_ab)
    test_r_mae_ab = np.mean(np.abs(test_rotations_ab_pred_euler - npmat2euler(test_rotations_ab)))
    test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)
    test_t_rmse_ab = np.sqrt(test_t_mse_ab)
    test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))



    print('==FINAL TEST==')
    print('A--------->B')
    print('EPOCH:: %s, rot_MSE: %f, rot_RMSE: %f, '
          'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
          % (str(epoch), test_r_mse_ab, test_r_rmse_ab, test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))

if __name__ == '__main__':
    args = get_args()
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    textio = IOStream('checkpoints/' + args.exp_name + '/log')
    with torch.no_grad():
        eval(args, 'best')

