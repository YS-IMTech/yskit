
import cv2

from PIL import Image
from pathlib import Path
import torch
import numpy as np



def make_path(path: Path) -> Path:
    path.mkdir(exist_ok=True, parents=True)
    return path

def rgb2mask(image): #input:tensor :mask_rgb [1,3,w,w]
        mask = torch.sum(image.squeeze(0),dim=0)  #[w,w]
        mask = (mask > 0).float().unsqueeze(0).unsqueeze(0)#[1,1,w,w]
        return mask

def tensor2Image(data, format='CHW'): 
    """
    Input
        data: type[torch.Tensor]
        format: 'CHW','HWC'
    Return
        img: np.uint8([0-255], [H,W,C]) 
    """
    
    scale = 1 if max(data) > 1 else 255
    ndims = data.ndim
    if format=='CHW':
        img = img.squeeze(0) if ndims==4 else None ##[1,C,H,W] -> [C,H,W]
        img = img.permute(1,2,0).detach().cpu().numpy() # [C,H,W] -> [H,W,C]
        img = (img.copy() * scale).astype(np.uint8)
    
    else: ## HWC
        img = img.squeeze(0) if ndims==4 else None ##[1,H,W,C] -> [H,W,C]
        img = img.detach().cpu().numpy()
        img = (img.copy() * scale).astype(np.uint8)

    return img 
    
def Image2tensor(data, format='CHW'): 
    """
    Input
        data
        format: Output tensor format-['CHW','HWC']
    Return
        tensor: value[0-1]
    """
    if format == 'CHW':
        tensor = torch.from_numpy(np.array(data)).permute(2,0,1).float() / 255.0 #[HWC]->[CHW]
    else: # HWC
        tensor = torch.from_numpy(np.array(data)).float() / 255.0
    return tensor



def save_img(path, img, format='HWC'):
    """
    Input:
        path: save path
        img: type[np.darray, tensor]
        format: input img type ('HWC','CHW')
    """
    scale = 1 if max(img) > 1 else 255
    ndims = img.ndim
    img = img.squeeze(0) if ndims==4 else None
    img = img.detach().cpu()
    
    if isinstance(img, np.ndarray):
        img = img.transpose(1,2,0) if format=='CHW' else None
        pil_img = Image.fromarray((img * scale).astype(np.uint8))
        pil_img.save(path)
        
    elif isinstance(img, torch.Tensor):

        img = img.permute(1,2,0) if format=='CHW' else None
        img = img.numpy()
        pil_img = Image.fromarray((img * scale).astype(np.uint8))
        pil_img.save(path)
        
    else:
        raise ValueError(f"Not Support this type: {type(img)}")
    

def read_img(path, mode="float", order="RGB"):

    """
    [NOTES] The input size unchanged [H, W, 1/3/4]
    Input:
        mode:
            pil
            float
            tensor/torch
            uint8
        order:
            RGB(default)

    Returns:
        img
    """
    if mode == "pil":
        return Image.open(path).convert(order)

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # cvtColor
    if order == "RGB":
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # mode
    if "float" in mode:
        return img.astype(np.float32) / 255
    elif "tensor" in mode or "torch" in mode:
        return torch.from_numpy(img.astype(np.float32) / 255)
    elif "uint8" in mode:
        return img
    else:
        raise ValueError(f"Unknown read_image mode {mode}")




