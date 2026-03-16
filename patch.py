import torchvision
import site
import os

print("torchvision:", torchvision.__version__)

# find python site-packages
site_packages = site.getsitepackages()[0]

deg_path = os.path.join(
    site_packages,
    "basicsr",
    "data",
    "degradations.py"
)

print("Looking for:", deg_path)

with open(deg_path, "r") as f:
    src = f.read()

old = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
new = "from torchvision.transforms.functional import rgb_to_grayscale"

if old in src:
    src = src.replace(old, new)

    with open(deg_path, "w") as f:
        f.write(src)

    print("Patched degradations.py ✓")

else:
    print("Already patched or not needed")
