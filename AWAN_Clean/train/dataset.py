import random
import h5py
import numpy as np
import torch
import torch.utils.data as udata
from torchvision.transforms import v2

import glob
import os

class BackgroundDataset(udata.Dataset):
    def __init__(
        self,
        config = None,
        df = None,
        train: bool = True,
    ) -> None:
        self.df = df.reset_index()
        self.df = self.df
        self.train = train
        self.crop_size = tuple(config.data.crop_size)
        self.out_channels = config.data.out_channels
        self.in_channels = config.data.in_channels

        mean = np.array(config.data.image_means)[self.in_channels]
        std = np.array(config.data.image_stds)[self.in_channels]

        mean = np.concatenate([
            mean, 
            np.array(config.data.image_means)[self.out_channels]
        ])
        std = np.concatenate([
            std, 
            np.array(config.data.image_stds)[self.out_channels]
        ])

        self.normalize = v2.Normalize(mean=mean, std=std)
        
        # self.transform = v2.Compose([
        #     # v2.ToTensor(),
        #     v2.ToImage(), 
        #     v2.ToDtype(torch.float32, scale=True),
        #     v2.RandomCrop(config.data.crop_size),
        # ])
        if self.train:
            self.transform = v2.Compose([
                # v2.ToTensor(),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomCrop(config.data.crop_size),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
            ])
        else:
            self.transform = v2.Compose([
                # v2.ToTensor(),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.CenterCrop(config.data.crop_size),
            ])

    # @staticmethod
    # def load_sample_from_path(path: str) -> tuple[np.array, np.array, np.array]:
    #     img = np.load(f"{path}/all_bands_target.npy")
    #     background_target = np.load(f"{path}/background_target.npy")
    #     ratios_ts_target = np.load(f"{path}/ratios_ts_target.npy")
    #     return img, np.exp(ratios_ts_target), np.exp(background_target)

    @staticmethod
    def load_sample(index_in_dataset):
        file_path = "/home/danielk-gpu/Projects/Momentick/momentick-monorepo/libs/train/tmp/updated_background_dataset.h5"
        with h5py.File(file_path, "r") as f:
            img = f[f'img_{index_in_dataset}'][:]
            # ratios_ts_target = f[f'ratios_ts_target_{index_in_dataset}'][:]
            # background_target = f[f'background_target_{index_in_dataset}'][:]

        return img

    def __len__(self) -> int:
        return self.df.shape[0]


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.df.loc[idx, 'index_in_dataset']
        target_range = len(self.out_channels)

        # img, target_background, background = self.load_sample(path)
        img = self.load_sample(path)
        target = img[..., self.out_channels]
        img = img[..., self.in_channels]

        fused = np.concatenate([img, target], axis=2)

        if self.transform:
            fused = self.transform(fused)

        if self.normalize:
            fused = self.normalize(fused)

        x, t = fused[:-target_range], fused[-target_range:]

        return x, t

    @staticmethod
    def compute_metrics(
        target: torch.Tensor, preds: torch.Tensor
    ) -> dict:

        metrics = {
            "mse": torch.nn.functional.mse_loss,
            }

        return {k: f(preds, target) for k, f in metrics.items()}

class HyperDatasetValid(udata.Dataset):
    def __init__(self, mode='valid'):
        if mode != 'valid':
            raise Exception("Invalid mode!", mode)
        data_path = './Dataset/Valid'
        data_names = glob.glob(os.path.join(data_path, '*.mat'))

        self.keys = data_names
        self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['cube']))
        hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)
        rgb = np.float32(np.array(mat['rgb']))
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb = torch.Tensor(rgb)
        mat.close()
        return rgb, hyper


class HyperDatasetTrain1(udata.Dataset):
    def __init__(self, mode='train'):
        if mode != 'train':
            raise Exception("Invalid mode!", mode)
        data_path = './Dataset/Train1'
        data_names = glob.glob(os.path.join(data_path, '*.mat'))

        self.keys = data_names
        random.shuffle(self.keys)
        # self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['cube']))
        hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)
        rgb = np.float32(np.array(mat['rgb']))
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb = torch.Tensor(rgb)
        mat.close()
        return rgb, hyper


class HyperDatasetTrain2(udata.Dataset):
    def __init__(self, mode='train'):
        if mode != 'train':
            raise Exception("Invalid mode!", mode)
        data_path = './Dataset/Train2'
        data_names = glob.glob(os.path.join(data_path, '*.mat'))

        self.keys = data_names
        random.shuffle(self.keys)
        # self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['cube']))
        hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)
        rgb = np.float32(np.array(mat['rgb']))
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb = torch.Tensor(rgb)
        mat.close()
        return rgb, hyper


class HyperDatasetTrain3(udata.Dataset):
    def __init__(self, mode='train'):
        if mode != 'train':
            raise Exception("Invalid mode!", mode)
        data_path = './Dataset/Train3'
        data_names = glob.glob(os.path.join(data_path, '*.mat'))

        self.keys = data_names
        random.shuffle(self.keys)
        # self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['cube']))
        hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)
        rgb = np.float32(np.array(mat['rgb']))
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb = torch.Tensor(rgb)
        mat.close()
        return rgb, hyper


class HyperDatasetTrain4(udata.Dataset):
    def __init__(self, mode='train'):
        if mode != 'train':
            raise Exception("Invalid mode!", mode)
        data_path = './Dataset/Train4'
        data_names = glob.glob(os.path.join(data_path, '*.mat'))

        self.keys = data_names
        random.shuffle(self.keys)
        # self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['cube']))
        hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)
        rgb = np.float32(np.array(mat['rgb']))
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb = torch.Tensor(rgb)
        mat.close()
        return rgb, hyper
