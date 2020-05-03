#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 23:24:47 2020

@author: ddd
"""

import numpy as np
import sys
sys.path.append('../')

from watercolor02_np_fft import to_water_collor, my_img_show

from PIL import Image
import matplotlib.pyplot as plt

if __name__ == '__main__': 
    img_dir = '../dataset/rand_abstr2'    
    tar_dir = img_dir+"annot"
    
    x = 4
    y = 6 
    first_im = 530   
    
    fig, axs = plt.subplots(nrows=x, ncols=y, subplot_kw={'xticks': [], 'yticks': []})
    
    dirs = ['rand_wc', 'rand_wcannot', 'rand_abstr2', 'rand_abstr2annot']
    
    for i in range(x):
        for j in range(y):
            image_file = f"../dataset/{dirs[i]}/{j+first_im}.png"
            img = plt.imread(image_file)
            axs[i][j].imshow(img)
    
    plt.tight_layout()
    plt.show()

#fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(9, 6),
#                        subplot_kw={'xticks': [], 'yticks': []})

#for ax, interp_method in zip(axs.flat, methods):
#    ax.imshow(grid, interpolation=interp_method, cmap='viridis')
#    ax.set_title(str(interp_method))




