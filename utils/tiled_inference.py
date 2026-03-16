import torch

@torch.no_grad()
def forward_tiled(model,x,tile=128,overlap=8):

    b,c,h,w=x.shape

    stride=tile-overlap

    out=torch.zeros_like(x)
    count=torch.zeros_like(x)

    for top in range(0,h,stride):
        for left in range(0,w,stride):

            t=min(top,h-tile)
            l=min(left,w-tile)

            patch=x[:,:,t:t+tile,l:l+tile]

            with torch.amp.autocast("cuda"):
                pred=model(patch).float()

            out[:,:,t:t+tile,l:l+tile]+=pred
            count[:,:,t:t+tile,l:l+tile]+=1

    return out/count.clamp(min=1)
