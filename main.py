

from __future__ import print_function
import os
import gc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import model
from data import ModelNet40
from util import transform_point_cloud, npmat2euler,cloudNearTrans,global_spacial_consistency
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import SHCNet
from chamfer_loss import ChamferLoss
from data_icl import TrainData, TestData
import util
import dataset

chamfer = ChamferLoss()
m = nn.AdaptiveMaxPool1d(1)

def cor_loss_compute(src, target, net, rotation_ab, translation_ab):
    st = transform_point_cloud(src, rotation_ab, translation_ab)
    src_corr, src_embedding, tgt_embedding, scores = net(src, target, True)
    rotation_ab_pred, translation_ab_pred = model.Registration(src, src_corr)
    loss_corr = util.correspondence_loss(st, target, scores)
    loss = loss_corr
    return loss, rotation_ab_pred, translation_ab_pred

def chamfer_loss_compute(src, target, net, rotation_ab, translation_ab):
    src_corr, src_embedding, tgt_embedding, scores = net(src, target, True)
    rotation_ab_pred, translation_ab_pred = model.Registration(src, src_corr)
    loss = chamfer(target.transpose(1, 2), src_corr.transpose(1,2))
    return loss, rotation_ab_pred, translation_ab_pred

def valid_one_epoch(args, net, test_loader):
    net.eval()

    total_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []

    for src, target, rotation_ab, translation_ab in tqdm(test_loader):
        src = src.cuda()
        target = target.cuda()
        rotation_ab = rotation_ab.cuda()
        translation_ab = translation_ab.cuda()

        batch_size = src.size(0)
        num_examples += batch_size

        if args.loss=='cor':
            loss, rotation_ab_pred, translation_ab_pred = cor_loss_compute(src, target, net, rotation_ab,
                                                                               translation_ab)
        elif args.loss=='chamfer':
            loss, rotation_ab_pred, translation_ab_pred = chamfer_loss_compute(src, target, net, rotation_ab, translation_ab)
        elif args.loss=='ang':
            src_corr, src_embedding, tgt_embedding, scores = net(src, target, True)
            rotation_ab_pred, translation_ab_pred = model.Registration(src, src_corr)
            identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
            loss = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                   + F.mse_loss(translation_ab_pred, translation_ab)

        total_loss += loss.item() * batch_size


        ## save rotation and translation
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())


    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)


    return total_loss * 1.0 / num_examples,\
           rotations_ab, translations_ab, rotations_ab_pred, translations_ab_pred

def train_one_epoch(args, net, train_loader, opt):
    net.train()

    total_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []


    for src, target, rotation_ab, translation_ab in tqdm(train_loader):
        src = src.cuda()
        target = target.cuda()
        rotation_ab = rotation_ab.cuda()
        translation_ab = translation_ab.cuda()

        batch_size = src.size(0)
        opt.zero_grad()
        num_examples += batch_size

        if args.loss=='cor':
            loss, rotation_ab_pred, translation_ab_pred = cor_loss_compute(src, target, net, rotation_ab,
                                                                               translation_ab)
        elif args.loss=='chamfer':
            loss, rotation_ab_pred, translation_ab_pred = chamfer_loss_compute(src, target, net, rotation_ab, translation_ab)
        elif args.loss=='ang':
            src_corr, src_embedding, tgt_embedding, scores = net(src, target, True)
            rotation_ab_pred, translation_ab_pred = model.Registration(src, src_corr)
            identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
            loss = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                   + F.mse_loss(translation_ab_pred, translation_ab)

        loss.backward()
        opt.step()


        total_loss += loss.item() * batch_size


        ## save rotation and translation
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())

    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    return total_loss * 1.0 / num_examples,\
           rotations_ab, translations_ab, rotations_ab_pred, translations_ab_pred

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

def train(args, train_loader, test_loader, textio):
    best_test_loss = np.inf

    best_test_r_mse_ab = np.inf
    best_test_r_rmse_ab = np.inf
    best_test_r_mae_ab = np.inf
    best_test_t_mse_ab = np.inf
    best_test_t_rmse_ab = np.inf
    best_test_t_mae_ab = np.inf

    net = model.Corresponodences(args).cuda()
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.epochs == 50:
        scheduler = MultiStepLR(opt, milestones=[20, 35, 45], gamma=0.1)
    else:
        scheduler = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1)

    for epoch in range(args.epochs):
        train_loss, train_rotations_ab, train_translations_ab, \
        train_rotations_ab_pred, train_translations_ab_pred = train_one_epoch(args, net, train_loader, opt)

        with torch.no_grad():
            test_loss, test_rotations_ab, test_translations_ab, \
            test_rotations_ab_pred,  test_translations_ab_pred= valid_one_epoch(args, net, test_loader)
        scheduler.step()

        train_rotations_ab_pred_euler = npmat2euler(train_rotations_ab_pred)
        train_r_mse_ab = np.mean((train_rotations_ab_pred_euler - npmat2euler(train_rotations_ab)) ** 2)
        train_r_rmse_ab = np.sqrt(train_r_mse_ab)
        train_r_mae_ab = np.mean(np.abs(train_rotations_ab_pred_euler - npmat2euler(train_rotations_ab)))
        train_t_mse_ab = np.mean((train_translations_ab - train_translations_ab_pred) ** 2)
        train_t_rmse_ab = np.sqrt(train_t_mse_ab)
        train_t_mae_ab = np.mean(np.abs(train_translations_ab - train_translations_ab_pred))


        test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
        test_r_mse_ab = np.mean((test_rotations_ab_pred_euler - npmat2euler(test_rotations_ab)) ** 2)
        test_r_rmse_ab = np.sqrt(test_r_mse_ab)
        test_r_mae_ab = np.mean(np.abs(test_rotations_ab_pred_euler - npmat2euler(test_rotations_ab)))
        test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)
        test_t_rmse_ab = np.sqrt(test_t_mse_ab)
        test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))


        if best_test_loss >= test_loss:
            best_test_loss = test_loss

            best_test_r_mse_ab = test_r_mse_ab
            best_test_r_rmse_ab = test_r_rmse_ab
            best_test_r_mae_ab = test_r_mae_ab

            best_test_t_mse_ab = test_t_mse_ab
            best_test_t_rmse_ab = test_t_rmse_ab
            best_test_t_mae_ab = test_t_mae_ab
            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
            else:
                torch.save(net.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)

        textio.cprint('==TRAIN==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f,rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, train_loss, train_r_mse_ab,
                         train_r_rmse_ab, train_r_mae_ab, train_t_mse_ab, train_t_rmse_ab, train_t_mae_ab))


        textio.cprint('==TEST==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f,rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, test_loss, test_r_mse_ab,
                         test_r_rmse_ab, test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))

        textio.cprint('==BEST TEST==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f,rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, best_test_loss,
                         best_test_r_mse_ab, best_test_r_rmse_ab,
                         best_test_r_mae_ab, best_test_t_mse_ab, best_test_t_rmse_ab, best_test_t_mae_ab))

        if torch.cuda.device_count() > 1:
            torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        else:
            torch.save(net.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        gc.collect()

def eval(args, epoch):
    net = SHCNet(args).cuda()
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.eval()
    if args.dataset == 'modelnet40':
        test_loader = DataLoader(
        ModelNet40(num_points=args.num_points, partition='test', gaussian_noise=args.gaussian_noise,
                   unseen=args.unseen, factor=args.factor, sub_points=args.sub_points),
        batch_size=1, shuffle=False, drop_last=False)
    elif args.dataset == 'icl_nuim':
        test_data = TestData(args)
        test_loader = DataLoader(test_data, 1)
    elif args.dataset == '7scenes':
        trainset, testset = dataset.get_datasets(args)
        test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=1, shuffle=False)
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

    print('==Load Optimation Modules==')
    print('A--------->B')
    print('EPOCH:: %s, rot_MSE: %f, rot_RMSE: %f, '
          'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
          % (str(epoch), test_r_mse_ab, test_r_rmse_ab, test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()

def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--emb_nn', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Embedding nn to use, [pointnet, dgcnn]')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--batch_size', type=int, default=4, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=4, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true', default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1234)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
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
    parser.add_argument('--loss', type=str, default='cor',
                        choices=['cor', 'chamfer', 'ang'], metavar='N',
                        help='dataset to use')

    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    _init_(args)

    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))

    if args.dataset == 'modelnet40':
        train_loader = DataLoader(
            ModelNet40(num_points=args.num_points, partition='train', gaussian_noise=False,
                       unseen=True, factor=args.factor, sub_points=1024),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            ModelNet40(num_points=args.num_points, partition='test', gaussian_noise=False,
                       unseen=True, factor=args.factor, sub_points=1024),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    elif args.dataset == 'icl_nuim':
        train_data = TrainData(args)
        train_loader = DataLoader(train_data, args.batch_size, drop_last=True, shuffle=True)
        test_data = TestData(args)
        test_loader = DataLoader(test_data, args.test_batch_size)
    elif args.dataset == '7scenes':
        trainset, testset = dataset.get_datasets(args)
        test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size, shuffle=False)
        train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size, shuffle=True)
    else:
        raise Exception("not implemented")


    if args.eval:
        eval(args, 'best')
    else:
        train(args, train_loader, test_loader, textio)
        eval(args, 'best')

    print('FINISH')

if __name__ == '__main__':
    main()
