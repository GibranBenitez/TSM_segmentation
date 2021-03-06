# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

# Notice that this file has been modified to support ensemble testing

import argparse
import time
import glob
import os

import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops import dataset_config
from torch.nn import functional as F

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# options
parser = argparse.ArgumentParser(description="TSM testing on the full validation set")
parser.add_argument('--dataset', type=str, default="biovid")

# may contain splits
parser.add_argument('--weights', type=str, default="./checkpoint/TSM_biovid2_RGB_resnet50_shift8_blockres_avg_segment16_e50_lr1e-04_val")
parser.add_argument('--val_ids', type=str, default="0,86")
parser.add_argument('--test_segments', type=str, default="16")
parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample as I3D')
parser.add_argument('--twice_sample', default=False, action="store_true", help='use twice sample for ensemble')
parser.add_argument('--dense_window', default=False, action="store_true", help='use dense_window')
parser.add_argument('--full_sample', default=True, action="store_true", help='use full_sample')
parser.add_argument('--full_res', default=False, action="store_true",
                    help='use full resolution 256x256 for test as in Non-local I3D')

parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--coeff', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

# for true test
parser.add_argument('--test_list', type=str, default=None)
parser.add_argument('--csv_file', type=str, default=None)

parser.add_argument('--softmax', default=False, action="store_true", help='use softmax')

parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--img_feature_dim',type=int, default=256)
parser.add_argument('--num_set_segments',type=int, default=1,help='TODO: select multiply set of n-frames from a video')
parser.add_argument('--pretrain', type=str, default='imagenet')

args = parser.parse_args()

bio_ids = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
       34, 35, 36, 39, 40, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56,
       58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75,
       76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
         correct_k = correct[:k].view(-1).float().sum(0)
         res.append(correct_k.mul_(100.0 / batch_size))
    return res


def parse_shift_option_from_log_name(log_name):
    if 'shift' in log_name:                 # shift8_blockres
        strings = log_name.split('_')
        for i, s in enumerate(strings):
            if 'shift' in s:
                break
        return True, int(strings[i].replace('shift', '')), strings[i + 1]       # T, 8, blockres
    else:
        return False, None, None

def parse_biovid_from_log_name(log_name):
    # TSM_biovid2_RGB_resnet50_shift8_blockres_avg_segment16_e50_lr1e-04_val0_gp22
    strings = log_name.split('_')
    if 'biovid' in log_name: 
        for i, s in enumerate(strings):
            if 'biovid' in s:
                break
        ipn_no_class = int(strings[i].replace('biovid', ''))
        for i, s in enumerate(strings):
            if 'val' in s:
                break
        return ipn_no_class, int(strings[i].replace('val', ''))
    elif 'ipn' in log_name:
        for i, s in enumerate(strings):
            if 'ipn' in s:
                break
        return 15-int(strings[i].replace('ipn', '')), 0
    else:
        return 1, 0


def eval_video(video_data, net, this_test_segments, modality):
    net.eval()
    with torch.no_grad():
        i, data, label = video_data
        batch_size = label.numel()
        num_crop = args.test_crops
        if args.dense_sample:
            num_crop *= 10  # 10 clips for testing when using dense sample
        if args.full_sample or args.dense_window:
            num_crop *= 138//this_test_segments + 1
        if args.twice_sample:
            num_crop *= 2

        if modality in ['RGB-flo', 'RGB-seg']:
            length = 4
        elif modality == 'Flow':
            length = 10
        elif modality == 'RGBDiff':
            length = 18
        elif modality == 'RGB':
            length = 3
        else:
            raise ValueError("Unknown modality "+ modality)

        data_in = data.view(-1, length, data.size(2), data.size(3))
        if is_shift:
            data_in = data_in.view(batch_size * num_crop, this_test_segments, length, data_in.size(2), data_in.size(3))
        rst = net(data_in)
        rst = rst.reshape(batch_size, num_crop, -1).mean(1)

        if args.softmax:
            # take the softmax to normalize the output to probability
            rst = F.softmax(rst, dim=1)

        rst = rst.data.cpu().numpy().copy()

        if net.module.is_shift:
            rst = rst.reshape(batch_size, num_class)
        else:
            rst = rst.reshape((batch_size, -1, num_class)).mean(axis=1).reshape((batch_size, num_class))

        return i, rst, label


val_ids = args.val_ids.split(',')
output = []
top1_ = AverageMeter()
top5_ = AverageMeter()
weights_name = args.weights
print(val_ids)
# for val_id in range(int(val_ids[0]),int(val_ids[1])+1):
for val_id in bio_ids:
    if args.dataset == "biovid":
        weights_name = list(glob.iglob(args.weights+str(val_id)+"*"))[0] 
    weights_name = os.path.join(weights_name,"ckpt.best.pth.tar")
    print(weights_name)
    weights_list = [weights_name]
    test_segments_list = [int(s) for s in args.test_segments.split(',')]
    assert len(weights_list) == len(test_segments_list)
    if args.coeff is None:
        coeff_list = [1] * len(weights_list)
    else:
        coeff_list = [float(c) for c in args.coeff.split(',')]

    if args.test_list is not None:
        test_file_list = args.test_list.split(',')
    else:
        test_file_list = [None] * len(weights_list)


    data_iter_list = []
    net_list = []
    modality_list = []

    total_num = None
    for this_weights, this_test_segments, test_file in zip(weights_list, test_segments_list, test_file_list):
        is_shift, shift_div, shift_place = parse_shift_option_from_log_name(this_weights)
        if 'Flow' in this_weights:
            modality = 'Flow'
        elif 'RGB-seg' in this_weights:
            modality = 'RGB-seg'
        elif 'RGB-flo' in this_weights:
            modality = 'RGB-flo'
        elif 'RGB' in this_weights:
            modality = 'RGB'
        this_arch = this_weights.split('TSM_')[1].split('_')[2]
        ipn_no_class, bio_validation = parse_biovid_from_log_name(this_weights)
        modality_list.append(modality)
        num_class, categories, args.train_list, val_list, root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                modality, ipn_no_class, str(bio_validation))
        print('=> shift: {}, shift_div: {}, shift_place: {}'.format(is_shift, shift_div, shift_place))
        net = TSN(num_class, this_test_segments if is_shift else 1, modality,
                  base_model=this_arch,
                  consensus_type=args.crop_fusion_type,
                  img_feature_dim=args.img_feature_dim,
                  pretrain=args.pretrain,
                  is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
                  non_local='_nl' in this_weights,
                  )

        if 'tpool' in this_weights:
            from ops.temporal_shift import make_temporal_pool
            make_temporal_pool(net.base_model, this_test_segments)  # since DataParallel

        net.RGBD_mod()
        checkpoint = torch.load(this_weights)
        checkpoint = checkpoint['state_dict']

        # base_dict = {('base_model.' + k).replace('base_model.fc', 'new_fc'): v for k, v in list(checkpoint.items())}
        base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
        replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                        'base_model.classifier.bias': 'new_fc.bias',
                        }
        for k, v in replace_dict.items():
            if k in base_dict:
                base_dict[v] = base_dict.pop(k)

        net.load_state_dict(base_dict)

        input_size = net.scale_size if args.full_res else net.input_size
        if args.test_crops == 1:
            cropping = torchvision.transforms.Compose([
                GroupScale(net.scale_size),
                GroupCenterCrop(input_size),
            ])
        elif args.test_crops == 3:  # do not flip, so only 5 crops
            cropping = torchvision.transforms.Compose([
                GroupFullResSample(input_size, net.scale_size, flip=False)
            ])
        elif args.test_crops == 5:  # do not flip, so only 5 crops
            cropping = torchvision.transforms.Compose([
                GroupOverSample(input_size, net.scale_size, flip=False)
            ])
        elif args.test_crops == 10:
            cropping = torchvision.transforms.Compose([
                GroupOverSample(input_size, net.scale_size)
            ])
        else:
            raise ValueError("Only 1, 5, 10 crops are supported while we got {}".format(args.test_crops))

        test_file = test_file if test_file is not None else val_list

        data_loader = torch.utils.data.DataLoader(
                TSNDataSet(root_path, test_file, num_segments=this_test_segments,
                           new_length=1 if modality in ['RGB', 'RGB-flo', 'RGB-seg'] else 5,
                           modality=modality,
                           image_tmpl=prefix,
                           test_mode=True,
                           remove_missing=len(weights_list) == 1,
                           transform=torchvision.transforms.Compose([
                               cropping,
                               Stack(roll=(this_arch in ['BNInception', 'InceptionV3']),mask=(modality in ['RGB-flo', 'RGB-seg'])),
                               ToTorchFormatTensor(div=(this_arch not in ['BNInception', 'InceptionV3'])),
                               GroupNormalize(net.input_mean, net.input_std),
                           ]), dense_sample=args.dense_sample, twice_sample=args.twice_sample, 
                           dense_window=args.dense_window, full_sample=args.full_sample, ipn=args.dataset=='ipn', ipn_no_class=ipn_no_class),
                batch_size=args.batch_size, shuffle=False,
                # num_workers=args.workers, pin_memory=True,
        )

        if args.gpus is not None:
            devices = [args.gpus[i] for i in range(args.workers)]
        else:
            devices = list(range(args.workers))

        net = torch.nn.DataParallel(net.cuda())
        net.eval()

        data_gen = enumerate(data_loader)

        if total_num is None:
            total_num = len(data_loader.dataset)
        else:
            assert total_num == len(data_loader.dataset)

        data_iter_list.append(data_gen)
        net_list.append(net)

    proc_start_time = time.time()
    max_num = args.max_num if args.max_num > 0 else total_num

    top1 = AverageMeter()
    top5 = AverageMeter()

    for i, data_label_pairs in enumerate(zip(*data_iter_list)):
        with torch.no_grad():
            if i >= max_num:
                break
            this_rst_list = []
            this_label = None
            for n_seg, (_, (data, label)), net, modality in zip(test_segments_list, data_label_pairs, net_list, modality_list):
                rst = eval_video((i, data, label), net, n_seg, modality)
                this_rst_list.append(rst[1])
                this_label = label
            assert len(this_rst_list) == len(coeff_list)
            for i_coeff in range(len(this_rst_list)):
                this_rst_list[i_coeff] *= coeff_list[i_coeff]
            ensembled_predict = sum(this_rst_list) / len(this_rst_list)

            for p, g in zip(ensembled_predict, this_label.cpu().numpy()):
                output.append([p[None, ...], g])
            cnt_time = time.time() - proc_start_time
            if num_class < 5:
                prec = accuracy(torch.from_numpy(ensembled_predict), this_label, topk=(1,))
                prec1 = prec[0]
                prec5 = prec[0]
            else:
                prec1, prec5 = accuracy(torch.from_numpy(ensembled_predict), this_label, topk=(1, 5))
            top1.update(prec1.item(), this_label.numel())
            top5.update(prec5.item(), this_label.numel())
            top1_.update(prec1.item(), this_label.numel())
            top5_.update(prec5.item(), this_label.numel())
            if i % 20 == 0:
                print('video {} done, total {}/{}, average {:.3f} sec/video, '
                      'moving Prec@1 {:.3f} Prec@5 {:.3f}'.format(i * args.batch_size, i * args.batch_size, total_num,
                                                                  float(cnt_time) / (i+1) / args.batch_size, top1.avg, top5.avg))

video_pred = [np.argmax(x[0]) for x in output]
video_pred_top5 = [np.argsort(np.mean(x[0], axis=0).reshape(-1))[::-1][:5] for x in output]

video_labels = [x[1] for x in output]

cf = confusion_matrix(video_labels, video_pred).astype(float)

# np.save('cm.npy', cf)
cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

cls_acc = cls_hit / cls_cnt
# print(cls_acc)
upper = np.mean(np.max(cf, axis=1) / cls_cnt)
print('upper bound: {}'.format(upper))

print('-----Evaluation is finished------')
print('Class Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
print('Overall Prec@1 {:.02f}% Prec@5 {:.02f}%'.format(top1_.avg, top5_.avg))
print('{}'.format(cf))

if args.csv_file is not None:
    if args.full_res:
        args.csv_file += '_Rfull'
    if args.dense_sample:
        args.csv_file += '_Sdense'
    elif args.twice_sample:
        args.csv_file += '_Stwice'
    elif args.full_sample:
        args.csv_file += '_Sfull'
    elif args.dense_window:
        args.csv_file += '_Swinds'
    args.csv_file += '.csv'
    args.csv_file = os.path.join(args.weights.replace('checkpoint', 'log'), args.csv_file)
    with open(test_file) as f:
        vid_names = f.readlines()
    vid_names = [n.split(',') for n in vid_names]
    if ipn_no_class > 1:
        vid_names = [item for item in vid_names if int(item[2]) > ipn_no_class-1]
        print('{}, {}'.format(len(vid_names), len(video_pred)))
    assert len(vid_names) == len(video_pred)
    print('=> Writing result to csv file: {}\n\n'.format(args.csv_file))
    if args.dataset != 'somethingv2':  # only output top1
        with open(args.csv_file, 'w') as f:
            for n, pred, labl in zip(vid_names, video_pred, video_labels):
                f.write('{},{},{}, {}, {}, {}\n'.format(n[0],n[3],n[4], categories[labl], categories[pred], categories[labl]==categories[pred]))
    else:
        with open(args.csv_file, 'w') as f:
            for n, pred5 in zip(vid_names, video_pred_top5):
                fill = [n]
                for p in list(pred5):
                    fill.append(p)
                f.write('{};{};{};{};{};{}\n'.format(*fill))
    with open(args.csv_file, 'a') as f:
        f.write('--------Evaluation is finished---------\n')        
        f.write('Upper bound, {:.04f}\n'.format(upper*100))
        f.write('Class Accuracy, {:.04f}\n'.format(np.mean(cls_acc) * 100))
        f.write('Overall, Prec@1, {:.04f}, Prec@5, {:.04f}\n'.format(top1_.avg, top5_.avg))
        assert len(cls_acc) == len(categories)
        f.write('Per Class Acc:\n') 
        for n, acc in zip(categories, cls_acc):
            f.write('{:s}, {:.04f}\n'.format(n, acc*100))
        f.write('\n{}'.format(cf))



        