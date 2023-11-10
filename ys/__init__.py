import lazy_loader

# lazy import that equals:
# from . import op, vis, bg, mesh
# from .utils import *
# from .env import env

__getattr__, __dir__, _ = lazy_loader.attach(
    __name__,
    submodules=["op","vis"],
    submod_attrs={
        "utils": ["make_path", "rgb2mask", "tensor2Image", "Image2tensor", "save_img"],
        "env": ["env"],
    },
)
