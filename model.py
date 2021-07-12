import torch
from torch import nn
import torch.nn.functional as F
import pdb


class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.conv = nn.Conv2d(1024, 256, 1)
        self.gcn = GCN(opt, 256, 256)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(-1, 1024, 1, 1)
        x = self.conv(x)
        # x = torch.tanh(x)
        x = F.relu(x)
        x = x.view(-1, 256)
        A = self.gcn(x)
        x = self.fc(x)
        mask = torch.sigmoid(x) + 1e-5
        inverse_mask = torch.reciprocal(mask)
        return mask, inverse_mask


class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.fc = nn.Linear(1024, 1)

    def forward(self, x):
        return self.fc(x)


class Network(nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()
        self.attention = Attention(opt)
        self.classification = Classification()

    def forward(self, x):
        mask, inverse_mask = self.attention(x)
        video_feature = torch.sum(x * mask, dim=0, keepdim=True) / torch.sum(mask)
        video_score = self.classification(video_feature)
        inverse_video_feature = torch.sum(x * inverse_mask, dim=0, keepdim=True) / torch.sum(inverse_mask)
        inverse_video_score = self.classification(inverse_video_feature)
        segments_scores = self.classification(x)
        return video_score, inverse_video_score, mask, segments_scores


class GCN(nn.Module):
    def __init__(self, opt, in_channels, out_channels):
        super(GCN, self).__init__()
        self.opt = opt
        if self.opt.C:
            self.theta = nn.Linear(in_channels, in_channels)
            self.phi = nn.Linear(in_channels, in_channels)
        self.conv_d = nn.Linear(in_channels, out_channels)
        if opt.residual:
            self.down = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        t, c = x.size()
        A, M = self.generate_A(t, self.opt.width)
        M = M.detach()
        if self.opt.A:
            A = A.detach()
        else:
            A = 0.
        if self.opt.C:
            theta = self.theta(x)
            phi = self.phi(x)
            C = torch.mm(theta, phi.permute(1, 0))
            if self.opt.CM:
                tmp = torch.exp(C - torch.max(C*M, dim=-1, keepdim=True)[0]) * M
                A += tmp / tmp.sum(dim=-1, keepdim=True)
            else:
                A += F.softmax(C, dim=-1)
        if self.opt.residual:
            out = self.conv_d(torch.bmm(A, x.permute(0, 2, 1)).permute(0, 2, 1)) + self.down(x)
        else:
            out = self.conv_d(torch.mm(A, x))
        return out

    @staticmethod
    def generate_A(dim, width=3):
        A = torch.zeros(dim, dim, device='cuda', requires_grad=False)
        min_value = -(width - 1) // 2
        extent = [min_value+i for i in range(width)]
        for i in range(dim):
            for j in extent:
                if i+j >=0 and i+j <=dim-1:
                    A[i, i+j] = 1.
        M = A
        A = A/A.sum(dim=1, keepdim=True)
        return A, M




