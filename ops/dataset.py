# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch.utils.data as data

from PIL import Image
import os
import numpy as np
from numpy.random import randint
# intesities = ['BL1', 'PA1', 'PA2', 'PA3', 'PA4']
dict_biovid = [None, None, 
        {'BL1': 0, 'PA4': 1}, {'BL1': 0, 'PA3': 1, 'PA4': 2},
        {'BL1': 0, 'PA2': 1, 'PA3': 2, 'PA4': 3},
        {'BL1': 0, 'PA1': 1, 'PA2': 2, 'PA3': 3, 'PA4': 4}]

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

class VideoRecordIPN(object):
    def __init__(self, row, idn=1):
        self._data = row
        self._idn = idn

    @property
    def path(self):
        return self._data[0]

    @property
    def st_frame(self):
        return int(self._data[3])

    @property
    def en_frame(self):
        return int(self._data[4])

    @property
    def num_frames(self):
        return int(self._data[-1])

    @property
    def label(self):
        return int(self._data[2])-int(self._idn)       


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False, remove_missing=False,
                 dense_sample=False, twice_sample=False, ipn=False, ipn_no_class=1):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        self.ipn = ipn
        self.id_noc = ipn_no_class
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            if self.image_tmpl == '{}_{:06d}.jpg':
                file_name = self.image_tmpl.format(directory, idx)
                return [Image.open(os.path.join(self.root_path, directory, file_name)).convert('RGB')]
            elif self.image_tmpl == '{}/{}_{:04d}.jpg':
                file_name = self.image_tmpl.format(directory, directory, idx)
                return [Image.open(os.path.join(self.root_path, directory.split('-')[0], file_name)).convert('RGB')]
            else:
                try:
                    return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
                except Exception:
                    print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                    return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
        elif self.modality == 'Flow':
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':  # ucf
                x_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('x', idx))).convert(
                    'L')
                y_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('y', idx))).convert(
                    'L')
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':  # something v1 flow
                x_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'x', idx))).convert('L')
                y_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'y', idx))).convert('L')
            else:
                try:
                    # idx_skip = 1 + (idx-1)*5
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert(
                        'RGB')
                except Exception:
                    print('error loading flow file:',
                          os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')
                # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
                flow_x, flow_y, _ = flow.split()
                x_img = flow_x.convert('L')
                y_img = flow_y.convert('L')
            return [x_img, y_img]
        elif self.modality in ['RGB-flo', 'RGB-seg']:
            if self.modality.split('-')[1] == 'flo':
                sensor = 'flow'
                ext = 'jpg'
            elif self.modality.split('-')[1] == 'seg':
                sensor = 'segment_five'
                ext = 'png'

            file_name = self.image_tmpl.format(directory, idx)
            imgC = Image.open(os.path.join(self.root_path, directory, file_name)).convert('RGB')
            imgD = Image.open(os.path.join(self.root_path.replace('frames',sensor), directory, file_name.replace('jpg',ext))).convert('RGB')
            if self.modality.split('-')[1] == 'seg':
                return [imgC, self._ipn_fassd(imgD.convert('L'))]
            else:
                return [imgC, imgD]

    def _ipn_fassd(self, img, pix_val=190):
        imgS = np.asarray(img)
        imgT = imgS.copy()
        imgT[imgS==pix_val] = 0
        imgT[imgT>0] = 255
        imgT = np.uint8(imgT)

        return Image.fromarray(np.concatenate([np.expand_dims(imgT, 2),np.expand_dims(imgT, 2),np.expand_dims(imgT, 2)], axis=2))

    def _parse_biovid(self, directory, frames=138):
        folder_list = os.listdir(os.path.join(self.root_path, directory))
        folder_list.sort()
        bio_labels = dict_biovid[self.id_noc]
        # print(list(bio_labels.keys()))
        out_list = []
        for i, path in enumerate(folder_list):
            if path.split('-')[1] not in list(bio_labels.keys()):
                continue
            out_list.append([path, frames, bio_labels[path.split('-')[1]]])

        return out_list

    def _parse_list(self):
        # hacer una funcion que genere todos los items de un folder de persona
        #   lo que generaria listas de forma que VideoRecord class se pueda usar
        # check the frame number is large >3:
        if self.ipn:
            tmp = [x.strip().split(',') for x in open(self.list_file)]
            if self.id_noc > 1:
                tmp = [item for item in tmp if int(item[2]) > self.id_noc-1]
            self.video_list = [VideoRecordIPN(item, self.id_noc) for item in tmp]
        elif self.image_tmpl == '{}/{}_{:04d}.jpg':
            val_id = int(self.list_file.split(',')[1])
            main_folder_list = os.listdir(self.root_path)
            main_folder_list.sort()
            if self.list_file.split(',')[0] == 'train':
                print('generating training list of {} subjects...'.format(len(main_folder_list)-1))
                tmp = []
                for item in main_folder_list:
                    if item != main_folder_list[val_id]:
                        tmp += self._parse_biovid(item)
            else:
                print('validating BioVid with subject: {}'.format(main_folder_list[val_id]))
                tmp = self._parse_biovid(main_folder_list[val_id])
            self.video_list = [VideoRecord(item) for item in tmp]
        else:
            tmp = [x.strip().split(' ') for x in open(self.list_file)]
            if not self.test_mode or self.remove_missing:
                tmp = [item for item in tmp if int(item[1]) >= 3]
            self.video_list = [VideoRecord(item) for item in tmp]

        if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            for v in self.video_list:
                v._data[1] = int(v._data[1]) / 2
        print('video clips:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:  # normal sample
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_val_indices(self, record):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_test_indices(self, record):
        if self.dense_sample:
            # # Orginal:
            # sample_pos = max(1, 1 + record.num_frames - 64)
            # t_stride = 64 // self.num_segments
            # start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            # Proposed:
            chunks = record.num_frames//self.num_segments
            t_stride = max(1, chunks // self.num_segments)
            sample_pos = max(1, 1 + record.num_frames - t_stride*self.num_segments)
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        elif self.twice_sample:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])

            return offsets + 1
        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder

        if self.image_tmpl == 'flow_{}_{:05d}.jpg':
            file_name = self.image_tmpl.format('x', 1)
            full_path = os.path.join(self.root_path, record.path, file_name)
        elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            file_name = self.image_tmpl.format(int(record.path), 'x', 1)
            full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
        elif self.image_tmpl == '{}_{:06d}.jpg':
            file_name = self.image_tmpl.format(record.path, record.st_frame)
            full_path = os.path.join(self.root_path, record.path, file_name)
        elif self.image_tmpl == '{}/{}_{:04d}.jpg':
            file_name = self.image_tmpl.format(record.path, record.path, 1)
            full_path = os.path.join(self.root_path, record.path.split('-')[0], file_name)
        else:
            file_name = self.image_tmpl.format(1)
            full_path = os.path.join(self.root_path, record.path, file_name)

        while not os.path.exists(full_path):
            print('################## Not Found:', os.path.join(self.root_path, record.path, file_name))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':
                file_name = self.image_tmpl.format('x', 1)
                full_path = os.path.join(self.root_path, record.path, file_name)
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
                file_name = self.image_tmpl.format(int(record.path), 'x', 1)
                full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
            elif self.image_tmpl == '{}_{:06d}.jpg':
                file_name = self.image_tmpl.format(record.path, record.st_frame)
                full_path = os.path.join(self.root_path, record.path, file_name)
            else:
                file_name = self.image_tmpl.format(1)
                full_path = os.path.join(self.root_path, record.path, file_name)

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):

        if self.ipn:
            stf = max(0, record.st_frame - 1)
            enf = record.en_frame
        else:
            stf = 0
            enf = record.num_frames

        images = list()
        for seg_ind in indices:
            p = int(seg_ind) + int(stf)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < enf:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)

if __name__ == '__main__':
    import torchvision
    from ops.transforms import *
    import pdb

    dataset = 'ipn'
    modality = 'RGB-seg'
    train_list = '/export/space0/gibran/dataset/HandGestures/IPN_dataset/IPNhand_TestList.txt'
    root_path = '/export/space0/gibran/dataset/HandGestures/IPN_dataset/frames'
    prefix = '{}_{:06d}.jpg'
    arch = 'resnet'
    crop_size = 224
    segments = 8

    datas = TSNDataSet(root_path, train_list, num_segments=segments,
               new_length=1,
               modality=modality,
               image_tmpl=prefix,
               transform=torchvision.transforms.Compose([
                   GroupCenterCrop(crop_size),
                   Stack(roll=(arch in ['BNInception', 'InceptionV3']),mask=(modality in ['RGB-flo', 'RGB-seg'])),
                   ToTorchFormatTensor(div=(arch not in ['BNInception', 'InceptionV3'])),
                   IdentityTransform(),
               ]), ipn=dataset=='ipn', ipn_no_class=1)

    vloader = torch.utils.data.DataLoader(
        TSNDataSet(root_path, train_list, num_segments=segments,
               new_length=1,
               modality=modality,
               image_tmpl=prefix,
               random_shift=False,
               transform=torchvision.transforms.Compose([
                   GroupCenterCrop(crop_size),
                   Stack(roll=(arch in ['BNInception', 'InceptionV3']),mask=(modality in ['RGB-flo', 'RGB-seg'])),
                   ToTorchFormatTensor(div=(arch not in ['BNInception', 'InceptionV3'])),
                   IdentityTransform(),
               ]), ipn=dataset=='ipn', ipn_no_class=1),
        batch_size=2, shuffle=False, pin_memory=True)

    pdb.set_trace()