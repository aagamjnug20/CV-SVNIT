import os
import glob
import numpy as np
import imageio.v3 as imageio
import torch
from torch.utils.data import Dataset
from PIL import Image


def add_noise_np(image_np,sigma):

    img=image_np.astype(np.float64)/255.0
    noise=np.random.normal(0,sigma/255.0,img.shape)

    return (img+noise)*255.0


def crop_to_multiple(image_np,s=8):

    h,w=image_np.shape[:2]

    return image_np[:h-h%s,:w-w%s]


def np_to_tensor(img):

    arr=np.clip(img,0,255).astype(np.float32)/255

    return torch.from_numpy(arr).permute(2,0,1)


class DIV2KDataset(Dataset):

    def __init__(self,dirs,patch_size=256,mode="train",sigma=50):

        if isinstance(dirs,str):
            dirs=[dirs]

        self.files=[]

        for d in dirs:
            self.files+=sorted(glob.glob(os.path.join(d,"*.png")))

        self.patch_size=patch_size
        self.mode=mode
        self.sigma=sigma

        assert len(self.files)>0

        if mode in ("val","test"):
            self.cache=[imageio.imread(f) for f in self.files]
        else:
            self.cache=None


    def __len__(self):
        return len(self.files)


    def __getitem__(self,idx):

        img_np=self.cache[idx] if self.cache is not None else imageio.imread(self.files[idx])

        if img_np.ndim==2:
            img_np=np.stack([img_np]*3,-1)

        if self.mode=="train":

            h,w=img_np.shape[:2]

            if h<self.patch_size or w<self.patch_size:

                img_np=np.array(Image.fromarray(img_np).resize(
                    (max(w,self.patch_size),max(h,self.patch_size)),Image.BICUBIC
                ))

            h,w=img_np.shape[:2]

            top=np.random.randint(0,h-self.patch_size+1)
            left=np.random.randint(0,w-self.patch_size+1)

            img_np=img_np[top:top+self.patch_size,left:left+self.patch_size]

            if np.random.rand()<0.5:
                img_np=np.fliplr(img_np)

            if np.random.rand()<0.5:
                img_np=np.flipud(img_np)

        else:

            img_np=crop_to_multiple(img_np)

        noisy_np=add_noise_np(img_np,self.sigma)

        return np_to_tensor(noisy_np),np_to_tensor(img_np)
