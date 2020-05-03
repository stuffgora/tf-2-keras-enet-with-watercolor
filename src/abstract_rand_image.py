#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 20:55:03 2020

@author: ddd
"""

import numpy as np, random
from PIL import Image
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from watercolor02_np_fft import to_water_collor


dX, dY = 256, 256 # 512, 512
xArray = np.linspace(0.0, 1.0, dX).reshape((1, dX, 1))
yArray = np.linspace(0.0, 1.0, dY).reshape((dY, 1, 1))

def randColor():
    return np.array([random.random(), random.random(), random.random()]).reshape((1, 1, 3))
def getX(): return xArray
def getY(): return yArray
def safeDivide(a, b):
    return np.divide(a, np.maximum(b, 0.001))

functions = [(0, randColor),
             (0, getX),
             (0, getY),
             (1, np.sin),
             (1, np.cos),
             (2, np.add),
             (2, np.subtract),
             (2, np.multiply),
             (2, safeDivide)]
depthMin = 4
depthMax = 20

def buildImg(depth = 0):
    funcs = [f for f in functions if
                (f[0] > 0 and depth < depthMax) or
                (f[0] == 0 and depth >= depthMin)]
    nArgs, func = random.choice(funcs)
    args = [buildImg(depth + 1) for n in range(nArgs)]
    return func(*args)

def my_img_show(img,title=""):
    #pass
    #return ""
    plt.figure()
    plt.imshow(img)
    plt.title(title)
    plt.show()
    
for i in range (1,2001):  
    pass
    img = buildImg()
    
    # Ensure it has the right dimensions, dX by dY by 3
    img = np.tile(img, (int(dX / img.shape[0]), int(dY / img.shape[1]), int(3 / img.shape[2])))
    

    odir = '../dataset/rand_abstr2'
    # Convert to 8-bit, send to PIL and save
    img8Bit = np.uint8(np.rint(img.clip(0.0, 1.0) * 255.0))
    #my_img_show(img8Bit,'srs')
    Image.fromarray(img8Bit).save(f'{odir}/{i}.png')
    y = to_water_collor(img8Bit.astype(np.float32)/255.0)
    #Rescale to 0-255 and convert to uint8
    y = (255.0 / y.max() * (y - y.min())).astype(np.uint8)
    #y = np.uint8(np.rint(y.clip(0.0, 1.0) * 255.0))
    Image.fromarray(img8Bit).save(f'{odir}annot/{i}.png')
    #my_img_show(y,'res')