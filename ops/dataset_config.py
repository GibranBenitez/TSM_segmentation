# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os

ROOT_DATASET = '/export/space0/gibran/dataset/'  # '/data/jilin/'


def return_ucf101(modality):
    filename_categories = 'UCF101/labels/classInd.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'UCF101/jpg'
        filename_imglist_train = 'UCF101/file_list/ucf101_rgb_train_split_1.txt'
        filename_imglist_val = 'UCF101/file_list/ucf101_rgb_val_split_1.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'UCF101/jpg'
        filename_imglist_train = 'UCF101/file_list/ucf101_flow_train_split_1.txt'
        filename_imglist_val = 'UCF101/file_list/ucf101_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51(modality):
    filename_categories = 51
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'HMDB51/images'
        filename_imglist_train = 'HMDB51/splits/hmdb51_rgb_train_split_1.txt'
        filename_imglist_val = 'HMDB51/splits/hmdb51_rgb_val_split_1.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'HMDB51/images'
        filename_imglist_train = 'HMDB51/splits/hmdb51_flow_train_split_1.txt'
        filename_imglist_val = 'HMDB51/splits/hmdb51_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_something(modality):
    filename_categories = 'something/v1/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1'
        filename_imglist_train = 'something/v1/train_videofolder.txt'
        filename_imglist_val = 'something/v1/val_videofolder.txt'
        prefix = '{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1-flow'
        filename_imglist_train = 'something/v1/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v1/val_videofolder_flow.txt'
        prefix = '{:06d}-{}_{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv2(modality):
    filename_categories = 'something-v2/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'something-v2/20bn-something-something-v2-frames'
        filename_imglist_train = 'something-v2/train_videofolder.txt'
        filename_imglist_val = 'something-v2/val_videofolder.txt'
        prefix = '{:06d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something-v2/20bn-something-something-v2-flow'
        filename_imglist_train = 'something-v2/train_videofolder_flow.txt'
        filename_imglist_val = 'something-v2/val_videofolder_flow.txt'
        prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_jester(modality):
    # filename_categories = 'jester-v1/category.txt'
    filename_categories = '20BN-jester/category.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        # root_data = ROOT_DATASET + 'jester-v1/20bn-jester-v1'
        # filename_imglist_train = 'jester-v1/train_videofolder.txt'
        # filename_imglist_val = 'jester-v1/val_videofolder.txt'
        root_data = ROOT_DATASET + '20BN-jester/20bn-jester-v1'
        filename_imglist_train = '20BN-jester/train_videofolder.txt'
        filename_imglist_val = '20BN-jester/val_videofolder.txt'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_ipn(modality):
    filename_categories = 'HandGestures/IPN_dataset/classIndAll.txt'
    if modality in ['RGB', 'RGB-flo', 'RGB-seg']:
        prefix = '{}_{:06d}.jpg'
        root_data = ROOT_DATASET + 'HandGestures/IPN_dataset/frames'
        filename_imglist_train = 'HandGestures/IPN_dataset/IPNhand_TrainList.txt'
        filename_imglist_val = 'HandGestures/IPN_dataset/IPNhand_TestList.txt'
    elif modality == 'Segment':
        prefix = '{}_{:06d}.jpg'
        root_data = ROOT_DATASET + 'HandGestures/IPN_dataset/segment'
        filename_imglist_train = 'HandGestures/IPN_dataset/IPNhand_TrainList.txt'
        filename_imglist_val = 'HandGestures/IPN_dataset/IPNhand_TestList.txt'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_kinetics(modality):
    filename_categories = 400
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'kinetics/images'
        filename_imglist_train = 'kinetics/labels/train_videofolder.txt'
        filename_imglist_val = 'kinetics/labels/val_videofolder.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_biovid(modality):
    filename_categories = 5
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'BioVid/PartA/crops'
        filename_imglist_train = 'train'
        filename_imglist_val = 'val'
        prefix = '{}/{}_{:04d}.jpg'  #071309_w_21-PA3-055_0069.jpg  071309_w_21/071309_w_21-PA3-055/
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_dataset(dataset, modality, ipn_no_class=1, bio_val='0'):
    dict_single = {'jester': return_jester, 'something': return_something, 'somethingv2': return_somethingv2,
                   'ucf101': return_ucf101, 'hmdb51': return_hmdb51, 'ipn': return_ipn,
                   'kinetics': return_kinetics, 'biovid': return_biovid}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    if isinstance(file_categories, str):        
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        if dataset == "ipn":
            categories = [item.rstrip().split(',')[1] for item in lines]
            for i in range(ipn_no_class): # removing classes of ipn dataset
                categories.remove(categories[0])
        else:
            categories = [item.rstrip() for item in lines]
    elif dataset == "biovid":
        categories = ["None"] * ipn_no_class
        file_imglist_train = 'train,{}'.format(bio_val)
        file_imglist_val = 'val,{}'.format(bio_val)
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, categories, file_imglist_train, file_imglist_val, root_data, prefix
