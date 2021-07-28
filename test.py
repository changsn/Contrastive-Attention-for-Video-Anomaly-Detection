from sklearn import metrics
import model
import numpy as np
import pickle
import torch
import argparse
import os
import input_data
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import xlsxwriter
import pdb
from scipy.interpolate import interp1d
from torch.autograd import Variable

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
    parser.add_argument('--B', action='store_false')
    parser.add_argument('--C', action='store_true')
    parser.add_argument('--BM', action='store_false')
    parser.add_argument('--CM', action='store_false')
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--num_gcn', default=1, type=int)
    parser.add_argument('--width', default=3, type=int)
    args = parser.parse_args()
    return args

def test(args):
    def draw_roc(tpr, fpr, auc):
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.4f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig("./test.png")
        plt.cla()
        plt.clf()
        plt.close()

    gts = np.load('./gts.npy',allow_pickle=True)
    test_features = test_data = input_data.InputData('./data/test', shuffle=False)
    net = model.Network(args).to('cuda')
    net.eval()
    best_auc = 0
    best_epoch = 0
    with torch.no_grad():
        for i in range(51, 200):
            workbook = xlsxwriter.Workbook('./record.xlsx')
            mask_sheet = workbook.add_worksheet('mask')
            score_sheet = workbook.add_worksheet('score')
            cell_format = workbook.add_format({'font_color': 'red'})
            cell_format2 = workbook.add_format({'font_color': 'blue'})
            net.load_state_dict(torch.load('./checkpoints/' + '{}.param'.format(i)))
            pred = []
            y = []
            for j in range(gts.shape[0]): 
                features = torch.from_numpy(test_features.next_batch(1)[0][0]).float().cuda()
                video_scores, inverse_video_scores, masks, segments_scores = net(Variable(features))
                # print mask
                row = np.squeeze(masks.cpu().numpy(), axis=1)
                mask_sheet.write_row(j, 1, row.tolist())
                mask_sheet.write(j, 0, np.mean(row), cell_format2)
                mask_sheet.conditional_format(j, np.argmax(row)+1, j, np.argmax(row)+1, {'type': 'no_errors', 'format': cell_format})
                # print score
                row = np.squeeze(segments_scores.cpu().numpy(), axis=1)
                score_sheet.write_row(j, 0, row.tolist())
                score_sheet.conditional_format(j, np.argmax(row), j, np.argmax(row), {'type': 'no_errors', 'format': cell_format})

                scores = np.squeeze(segments_scores.cpu().numpy())
                video_score = video_scores.cpu().numpy()
                if video_score[0, 0] < -2:
                    scores += video_score[0, 0]
				
                x = np.arange(0, scores.shape[0])
                f = interp1d(x, scores, kind='linear', axis=0, fill_value='extrapolate')
                scale_x = np.arange(0, scores.shape[0], 1 / 60)
                pred += list(f(scale_x))
                y += gts[j].tolist()

            fpr, tpr, thresholds = metrics.roc_curve(y, pred)
            auc = metrics.auc(fpr, tpr)
            draw_roc(tpr, fpr, auc)
            print('Epoch: {}, AUC: {}'.format(i, auc))
            if auc > best_auc:
                best_auc = auc
                best_epoch = i
            workbook.close()
    print('Best_Epoch: {}, Best_AUC: {}'.format(best_epoch, best_auc))
    return best_auc

if __name__ == '__main__':
    args = parse_args()
    print('Hyper-parameters:')
    d_args = vars(args)
    for i in d_args:
        print('{}: {}'.format(i, d_args[i]))
    test(args)
