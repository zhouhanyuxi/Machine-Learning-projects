from PIL import Image
import numpy as np

from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

import net
import sampler


def extractDeepFeature(img, model, is_gray):
    if is_gray:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5,), std=(0.5,))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
    img, img_ = transform(img), transform(F.hflip(img))
    img, img_ = img.unsqueeze(0).to('cuda'), img_.unsqueeze(0).to('cuda')
    ft = torch.cat((model(img), model(img_)), 1)[0].to('cpu')
    return ft


def KFold(n=6000, n_folds=10):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[int(i * n / n_folds):int((i + 1) * n / n_folds)]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


def eval(model, model_path=None, is_gray=False):
    predicts = []
    model.load_state_dict(torch.load(model_path))
    model.eval()
    #这里放一个sampler
    #-------------sampler---------------
    root = '/mnt/batch/tasks/shared/LS_root/mounts/clusters/yikai/code/Users/yili.lai/recognition/CUB_200_2011/test/'
    pairs_lines = sampler.get_pair_sample_list(root, '/mnt/batch/tasks/shared/LS_root/mounts/clusters/yikai/code/Users/yikai.wang', 6)
    #with open('/mnt/batch/tasks/shared/LS_root/mounts/clusters/yikai/code/Users/yikai.wang/pairs.txt') as f:
    #    pairs_lines = f.readlines()
    #---------------end-----------------
    print('eval begin...')

    with torch.no_grad():
        for i in range(1200):
            p = pairs_lines[i]
            name1 = p[0]
            name2 = p[1]
            sameflag = p[2]
            #p = pairs_lines[i].replace('\n', '').split('\t')

            #if 3 == len(p):
            #    sameflag = 1
            #    name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            #    name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
            #elif 4 == len(p):
            #    sameflag = 0
            #    name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            #    name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
            #else:
            #    raise ValueError("WRONG LINE IN 'pairs.txt! ")

            with open(root + name1, 'rb') as f:
                img1 =  Image.open(f).convert('RGB')
                img1 = img1.resize((112,96),Image.BILINEAR)
            with open(root + name2, 'rb') as f:
                img2 =  Image.open(f).convert('RGB')
                img2 = img2.resize((112,96),Image.BILINEAR)
            f1 = extractDeepFeature(img1, model, is_gray)
            f2 = extractDeepFeature(img2, model, is_gray)

            distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            predicts.append('{}\t{}\t{}\t{}\n'.format(name1, name2, distance, sameflag))

    accuracy = []
    thd = []
    folds = KFold(n=1200, n_folds=10)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    #predicts = np.array(map(lambda line: line.strip('\n').split(), predicts))
    predicts = np.array([x.replace('\n', '').split('\t') for x in predicts])
    for idx, (train, test) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)
    print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))

    return np.mean(accuracy), predicts


if __name__ == '__main__':
    _, result = eval(net.sphere().to('cuda'), model_path='checkpoint/CosFace_24_checkpoint.pth')
    np.savetxt("result/result.txt", result, '%s')
