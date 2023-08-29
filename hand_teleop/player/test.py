import os
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import imageio

from torchvision.utils import save_image
from torchvision.transforms import v2

real_images = '/real/raw_data/dclaw/0000'
real_images = '/real/raw_data/pick_place_mustard_bottle/0000'
idx = 0
rgb_pic = torchvision.io.read_image(path = os.path.join('.'+real_images, "frame%04i.png" % idx), mode=torchvision.io.ImageReadMode.RGB)

padded_imgs = v2.Pad(padding=[0,80])(rgb_pic)
padded_imgs = v2.Resize(size=[224,224])(padded_imgs)
padded_imgs = padded_imgs.permute(1,2,0)
padded_imgs = padded_imgs.cpu().detach().numpy()
#rgb_pic = (rgb_pic / 255.0).type(torch.float32)
# padded_imgs.type(torch.uint8)
print(padded_imgs.dtype) 
print(padded_imgs.shape) 
imageio.imsave('./real/test/0000.png',padded_imgs)
