import os
import glob
import numpy as np
import imageio.v3 as imageio
import torch
from torch.utils.data import Dataset
from PIL import Image


def add_noise_np(image_np, sigma):

    img = image_np.astype(np.float32)/255
    noise = np.random.normal(0,sigma/255,img.shape)

    return (img+noise)*255


def np_to_tensor(img):

    img = np.clip(img,0,255).astype(np.float32)/255

    return torch.from_numpy(img).permute(2,0,1)


class DenoiseDataset(Dataset):

    def __init__(self, dirs, patch_size=128, sigma=50, mode="train"):

        self.files=[]

        for d in dirs:
            self.files+=glob.glob(os.path.join(d,"*.png"))

        self.patch=patch_size
        self.sigma=sigma
        self.mode=mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):

        img=imageio.imread(self.files[idx])

        if img.ndim==2:
            img=np.stack([img]*3,-1)

        if self.mode=="train":

            h,w=img.shape[:2]

            top=np.random.randint(0,h-self.patch)
            left=np.random.randint(0,w-self.patch)

            img=img[top:top+self.patch,left:left+self.patch]

        noisy=add_noise_np(img,self.sigma)

        return np_to_tensor(noisy),np_to_tensor(img)
