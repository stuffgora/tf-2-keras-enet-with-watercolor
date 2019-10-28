#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 23:54:04 2019

@author: ddd
"""

import argparse
import pathlib
import matplotlib.pyplot as plt
# example of calculating the frechet inception distance
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm

# calculate frechet inception distance
def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

def test():
    # define two collections of activations
    m = 512
    n = 512
    act1 = random(m*n)
    act1 = act1.reshape((m,n))
    act2 = random(m*n)
    act2 = act2.reshape((m,n))
    # fid between act1 and act1
    fid = calculate_fid(act1, act1)
    print('FID (same): %.3f' % fid)
    # fid between act1 and act2
    fid = calculate_fid(act1, act2)
    print('FID (different): %.3f' % fid)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(f"--ref_dir", type=str, default=None, help="dir containing ref images")
    parser.add_argument(f"--dir", type=str, default=None, help="dir to compate images from it")
    
    cfg = parser.parse_args()
    
    cfg.ref_dir = 'dataset/testannot' 
    cfg.dir = 'models/camvid/enet_no_wc/visualization'
    
    dir1= pathlib.Path(cfg.ref_dir) 
    dir2 = pathlib.Path(cfg.dir)
    
    flst1 = [str(f) for f in dir1.glob('*.png')][:20]
    flst2 = [str(f) for f in dir2.glob('*.png')][:20]
    
    fig, ((ax1,ax2)) = plt.subplots(1,2)
    for ax in [ax1,ax2]: ax.axis('off')
    #plt.figure(figsize=(10,10))
    img1 = plt.imread(flst1[0])
    img2 = plt.imread(flst2[0])
    ax1.imshow(img1) #,cmap='winter')
    ax2.imshow(img2)
    plt.show()
    