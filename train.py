import torch
from torch.utils.data import DataLoader

from models.ensemble import TripleEnsemble,load_restormer,load_nafnet,load_model_c
from data.dataset import DenoiseDataset
from utils.losses import combined_loss
from utils.metrics import psnr
from utils.tiled_inference import forward_tiled


DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DIRS=["datasets/train"]
VAL_DIRS=["datasets/val"]

NAF_CKPT="checkpoints/nafnet.pth"
REST_CKPT="checkpoints/restormer.pth"
MODEL_C_CKPT="checkpoints/nafnet_w32.pth"

EPOCHS=100
BATCH=2


def train_epoch(model,loader,optim):

    model.train()

    for noisy,clean in loader:

        noisy=noisy.to(DEVICE)
        clean=clean.to(DEVICE)

        pred=model(noisy)

        loss=combined_loss(pred,clean)

        optim.zero_grad()
        loss.backward()
        optim.step()


@torch.no_grad()
def validate(model,loader):

    model.eval()

    total_psnr=0

    for noisy,clean in loader:

        noisy=noisy.to(DEVICE)
        clean=clean.to(DEVICE)

        pred=forward_tiled(model,noisy)

        total_psnr+=psnr(pred,clean)

    return total_psnr/len(loader)


def main():

    restormer=load_restormer(REST_CKPT)
    nafnet=load_nafnet(NAF_CKPT)
    model_c=load_model_c(MODEL_C_CKPT)

    model=TripleEnsemble(restormer,nafnet,model_c).to(DEVICE)

    train_ds=DenoiseDataset(TRAIN_DIRS)
    val_ds=DenoiseDataset(VAL_DIRS,mode="val")

    train_loader=DataLoader(train_ds,batch_size=BATCH,shuffle=True)
    val_loader=DataLoader(val_ds,batch_size=1)

    optim=torch.optim.AdamW(model.parameters(),lr=1e-4)

    best_psnr=0

    for epoch in range(EPOCHS):

        train_epoch(model,train_loader,optim)

        val_psnr=validate(model,val_loader)

        print("epoch",epoch,"psnr",val_psnr)

        if val_psnr>best_psnr:

            best_psnr=val_psnr

            torch.save(
                {"model":model.state_dict()},
                "checkpoints/best.pth"
            )


if __name__=="__main__":
    main()
