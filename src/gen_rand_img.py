# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 02:45:27 2020

@author: abraverm
"""

import numpy as np
from watercolor02_np_fft import to_water_collor, my_img_show

from PIL import Image

def rand_img(m,n):
#    img = '../stars_clean.png'    
#    x = np.array(Image.open(img).convert("RGB"))/256.0 
    
    x = np.random.rand(m,n,3)
    y = to_water_collor(x)
    yield (x,y)

if __name__ == '__main__': 
    size = 256
    for i in range(1):
        for im in rand_img(size,size):
            my_img_show(im[0], "src")
            my_img_show(im[1], "res")
        
    