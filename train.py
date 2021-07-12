import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import csv
import model
import input_data
import argparse
from test import test
import pdb

train_feature_code_path = './data/train'
test_feature_code_path = './data/test'


def parse_args():
    parser = argparse.ArgumentParser()
    # input for training
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--iterations', default=67, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.5e-3, type=float)
    parser.add_argument('--restore', default=False, type=bool)
    parser.add_argument('--sal_coe', default=0.5, type=float)
    parser.add_argument('--weight_decay', default=0.2e-5, type=float)
    parser.add_argument('--sal_ratio', default=0.3, type=float)
    parser.add_argument('--save_path', default='./checkpoints/', type=str)
    parser.add_argument('--gpu_list', default=[0], type=list)
    parser.add_argument('--TEST', default=True, type=bool)

    parser.add_argument('--A', action='store_false')
    parser.add_argument('--C', action='store_true')
    parser.add_argument('--CM', action='store_false')
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--num_gcn', default=1, type=int)
    parser.add_argument('--width', default=3, type=int)
    args = parser.parse_args()
    return args


def tower_loss(net, features, labels, dims, args):
    loss = []
    inverse_loss = []
    sum_sal_loss = []
    labels = torch.from_numpy(labels).cuda()
    for i in range(len(features)):
        feature = torch.from_numpy(features[i]).cuda()
        video_score, inverse_video_score, mask, seg_scores = net(feature)
        entropy_loss = F.binary_cross_entropy_with_logits(video_score, labels[i: i+1, :])
        margin = torch.max(torch.tensor(0., device='cuda', requires_grad=False), (torch.sigmoid(seg_scores) - mask) ** 2 - args.sal_ratio ** 2)
        count_nonzero = (margin != 0.).sum().detach().to(torch.float32)
        sal_loss = torch.sum(margin) / (count_nonzero + 1e-6)
        inverse_entropy_loss = labels[i, 0] * F.binary_cross_entropy_with_logits(inverse_video_score, torch.tensor([[0.]], requires_grad=False, device='cuda'))
        loss.append(entropy_loss)
        inverse_loss.append(inverse_entropy_loss + args.sal_coe * sal_loss)
        sum_sal_loss.append(args.sal_coe * sal_loss)
    return sum(loss) / args.batch_size, sum(inverse_loss) / args.batch_size, sum(sum_sal_loss) / args.batch_size


def train():
    args = parse_args()
    print('Hyper-parameters:')
    d_args = vars(args)
    for i in d_args:
        print('{}: {}'.format(i, d_args[i]))
    gpu_list = args.gpu_list
    num_gpus = len(gpu_list)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpu_list])
    net = model.Network(args)
    net.to('cuda')
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_ass = torch.optim.Adam(net.attention.parameters(), lr=args.lr)
    train_data = input_data.InputData(train_feature_code_path, shuffle=True)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    for i in range(args.epochs):
        print('[*] Current epochs: %d ---' % i)
        sum_loss = 0.
        sum_inverse_loss = 0.
        sum_sum_sal_loss = 0.
        for j in range(args.iterations):
            list_features, numpy_labels, numpy_dims = train_data.next_batch(size=args.batch_size)
            loss, inverse_loss, sum_sal_loss = tower_loss(net, list_features, numpy_labels, numpy_dims, args)
            optimizer.zero_grad()
            optimizer_ass.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            inverse_loss.backward()
            optimizer_ass.step()
            sum_loss += loss.item()
            sum_inverse_loss += inverse_loss.item()
            sum_sum_sal_loss += sum_sal_loss.item()
        print('Loss: {:.3f}, Inverse Loss: {:.3f}, sal_loss: {:.3f}'.format(sum_loss / args.iterations, sum_inverse_loss / args.iterations, sum_sum_sal_loss / args.iterations))
        if i > 50:
            torch.save(net.state_dict(), args.save_path + '{}.param'.format(i))
    if args.TEST:
        test(args)


if __name__ == '__main__':
    train()

