from scipy.io import loadmat,savemat
import torch
import glob
from torchvision import transforms
from torch.utils.data import Dataset
import os
import cv2
from utils import TensorNorm

class ProjectionsImage(Dataset):
    def __init__(self, data_path):

        self.projection_path = data_path

        self.imgs = glob.glob(os.path.join(self.projection_path, '*.tiff'))

    def __getitem__(self, index):

        img = cv2.imread(self.imgs[index], cv2.IMREAD_GRAYSCALE)

        tensor_img = transforms.ToTensor()(img)

        norm_img = transforms.Normalize(mean=torch.mean(tensor_img),
                                         std=torch.std(tensor_img))(tensor_img)

        return norm_img.to(dtype=torch.float32)

    def __len__(self):
        return len(self.imgs)


class Projections_Mat(Dataset):
    def __init__(self, mat_path):

        self.mat_path = mat_path

        self.mat = loadmat(self.mat_path)

    def __getitem__(self, index):

        img = self.mat['wzy_proj'][:,:,index]

        tensor_img = torch.tensor(img).unsqueeze(0)
        tensor_img = TensorNorm(tensor_img)

        norm_img = transforms.Normalize(mean=torch.mean(tensor_img),
                                   std=torch.std(tensor_img))(tensor_img)
        norm_img = norm_img.to(dtype=torch.float32)

        return norm_img.to(dtype=torch.float32)

    def __len__(self):
        return int(self.mat['wzy_proj'].shape[-1])