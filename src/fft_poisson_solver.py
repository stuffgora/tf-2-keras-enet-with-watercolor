# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 00:18:56 2019

@author: abraverm
"""

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft,fft2,ifft2,ifft,irfft2,rfft2
#import random as random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.image import imread
import math
#import cv2

#from scipy.fftpack import dst,idst as dst2, idst2
from scipy.fftpack import dst as dst_r
from scipy.fftpack import dct as dct


def dst(x,axis=-1):
    """Discrete Sine Transform (DST-I)

    Implemented using 2(N+1)-point FFT
    xsym = r_[0,x,0,-x[::-1]]
    DST = (-imag(fft(xsym))/2)[1:(N+1)]

    adjusted to work over an arbitrary axis for entire n-dim array
    """
    n = len(x.shape)
    N = x.shape[axis]
    slices = [None]*3
    for k in range(3):
        slices[k] = []
        for j in range(n):
            slices[k].append(slice(None))
    newshape = list(x.shape)
    newshape[axis] = 2*(N+1)
    xsym = np.zeros(newshape,np.float)
    slices[0][axis] = slice(1,N+1)
    slices[1][axis] = slice(N+2,None)
    slices[2][axis] = slice(None,None,-1)
    for k in range(3):
        slices[k] = tuple(slices[k])
    xsym[slices[0]] = x
    xsym[slices[1]] = -x[slices[2]]
    #print(-x[slices[2]])
    #my_imshow(xsym, 'xsym ')
    xsym = np.ndarray.astype(xsym,complex)
    DST = fft(xsym,axis=axis)
    
    #my_imshow(DST.imag, 'cmpx_dst img')
    #my_imshow(DST.real, 'cmpx_dst real')
    #print xtilde
    return (-(DST.imag)/2)[slices[0]]
    #return dct(xsym,type =2, axis=axis, n=N)


# Define 2 dimensional DST, the idst is same as DST because of symmetry https://arsenous.wordpress.com/2013/03/22/221/
def dst2(x,axes=(-1,-2)):
    return dst(dst(x,axis=axes[0]),axis=axes[1])

def idst2(x,axes=(-1,-2)):
    return dst(dst(x,axis=axes[0]),axis=axes[1])
               

def fft_poisson(f,h=1):
    m,n=f.shape
    
    #m = 64#float(f.shape[0])
    #n = 64#float(f.shape[1])

    # -> f_bar=np.zeros([n,n])
    f_bar=np.zeros([m,n])
    u_bar = f_bar            # make of the same shape
    u = u_bar
    
    #my_imshow(f,'f')

    f_bar=idst2(f)            # f_bar= fourier transform of f
    #my_imshow(f_bar,'f_bar')
    #f_bar = f_bar * (2/n+1)**2  #Normalize
    f_bar = f_bar * (2.0/n+1)*(2.0/m+1) #Normalize
    #my_imshow(f_bar, 'f_bar normolized')
    #u_bar =np.zeros([n,n])
    pi=np.pi
    #lam = np.arange(1,n+1)
    #lam = -4/h**2 * (np.sin((lam*pi) / (2*(n + 1))))**2 #$compute $\lambda_x$
    
    lam_n = np.arange(1,n+1)
    lam_n = -4/h**2 * (np.sin((lam_n*pi) / (2*(n + 1))))**2 
    
    lam_m = np.arange(1,m+1)
    lam_m = -4/h**2 * (np.sin((lam_m*pi) / (2*(m + 1))))**2 #$compute $\lambda_x$
    
    #lam_mat = np.reshape(lam, [-1,1])
    #lam_mat = np.matmul(lam_mat , lam_mat.T)
    #my_imshow(lam_mat, 'lam_mat')
    #u_bar = np.divide( f_bar, lam_mat) 
    #for rectangular domain add $lambda_y$
    for i in range(0,m):
        for j in range(0,n):
            u_bar[i,j] = (f_bar[i,j]) / (lam_m[i] + lam_n[j])
    #u_bar = f_bar
    #my_imshow(u_bar,"u_bar")
    u=dst2(u_bar)                #sine transform back
    #my_imshow(u,"u")
    u = u * (2.0**2/((n+1)*(m+1)))            #normalize ,change for rectangular domain
    #my_imshow(u,"u normalized")
    return u

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def imgx(img):
    gx = np.zeros(img.shape) 
    #gx[j+1,k] = img[j+1,k] - gx[j,k]
    gx[1:,:,...] = img[1:,:,...] - img[:-1,:,...]
    return gx

def norm_0_1_rgb(a):
    #return (x-min(x))/(max(x)-min(x))
    return (a - np.min(a))/np.ptp(a)

def to_opponent(x):
    s2 = math.sqrt(2.0)
    s6 = math.sqrt(6.0)
    a = 0.2989
    b = 0.5780
    c = 0.1140
    oponent_transf = [[1/s2, -1/s2,  0   ],
                      [1/s6,  1/s6, -2/s6],
                      [  a ,    b ,  c   ]
                      ]
    return np.matmul(x, oponent_transf)

def my_imshow(img, title = ''):
    #pass
    #return ""
    plt.imshow(img)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    img_file = 'C:/Users/abraverm/Desktop/personal/univer/PycharmProjects/Img_Ilusions/im.png'

    from PIL import Image
    #from scipy.misc import imresize, imread
    im_w = 10
    im_h = 10
    im_pad = 2
    
    img = Image.open(img_file)
    img.thumbnail((im_w,im_h), Image.ANTIALIAS)
    img = np.array(img)/256
    
    #img = imread(img_file)
    #img = imresize(img,[65,64])
    
    #mask = np.zeros(img.shape)
    #img = rgb2gray(img)
    
    mask = img*1
    #mask[10:-10,10:-10,:] = 0
    mask[im_pad:-im_pad,im_pad:-im_pad] = 0
    img = img - mask
    #img = to_opponent(img)
#    plt.imshow(img)
#    plt.show()
    my_imshow(img,'img with mask')
    
    gx = imgx(img)
    gxx = imgx(gx)
    #plt.imshow(gx)
    #plt.show()
    my_imshow(gx,'gx')
    
    gy = imgx(img.T).T
    gyy = imgx(gy.T).T
    
    #poisson_solver(gx,gy,boundary_image=img)
    #h = dx; h= L/n
    h = 1
    u = np.zeros(img.shape)
    f =[]
    #img = np.expand_dims(img, axis=2)
    
    
    # %%
    
    im = img[:,:,0]/np.max(img[:,:,0])
    print(np.max(im))
    print(np.min(im))
    my_imshow(im,"src")
    im = dst(im)
    my_imshow(im,"dst")
    im = dst(im)
    my_imshow(im,"idst")
    # %%
    for c in range(1) : # img.shape[2]):
        i = img[:,:,c]
        gx = imgx(i)
        gxx = imgx(gx)
        gy = imgx(i.T).T
        gyy = imgx(gy.T).T
        
        #f = gxx[:,:,c] + gyy[:,:,c]
        f = gxx + gyy
        #U=norm_0_1_rgb(fft_poisson(f,h))
        U=fft_poisson(f,h)
        print(f.shape)
        print(U.shape)
        print(u.shape)
#        plt.imshow(U)
#        plt.show()
        my_imshow(U, f'U layer {c}')

        u[0:,:,c] = U
        
#        plt.imshow(img[:,:,c])
#        plt.show()
        my_imshow(img[:,:,c], f'img layer {c}')
    
    my_imshow(u,'u result')
    
    
#    plt.imshow(u)
#    plt.show()
    
    #plt.imshow(f)
    #plt.show()
    
    
##    set bounds a,b,parameters
#    a = 0; b = 1;    
#    alpha=10                #alpha is grid points=2^alpha
#    n=2**alpha
#    L=b-a                    #length of system
#    
#    xe =np.linspace(a,b,n); 
#    ye =np.linspace(a,b,n);
#    x, y = np.meshgrid(xe,ye)
#    h=L/(n);            #size 
#    h2 = h**2;            #h squared
#    hx2=h2;             #started with a cube,hx2=hy2
#    hy2=h2;
#    f=np.zeros([n,n]);
#    #initial conditions
#    
#    #f[round(n/2):n,round(n/2):n]=1; #Initial condition
#    f[round((n+1)/2),round((n+1)/2-10)]=20
#    #f[round((n+1)/2),round((n+1)/2+10)]=-20
#    f[random.randint(0,n),random.randint(0,n)]=-10    #put a random charge
#    
#    nx,ny=np.array(f.shape)-1         #last index
#    U=np.zeros([n,n])
#    
#    # BOUNDARY CONDITIONS
#    #set U(x,y)=g(x,y)
#    U[0,:]=0
#    U[nx,:]=0#5e-5
#    U[:,0]=0
#    U[:,ny]=0
#    
#    ##homogenize boundary condition
#    
#    f[1,:]=f[1,:]+U[0,:]/hx2;
#    f[nx-1,:]=f[nx-1,:]+U[n-1,:]/hx2;
#    
#    f[:,1]=f[:,1]+U[:,0]/hy2;
#    f[:,ny-1]=f[:,ny-1]+U[:,n-1]/hy2;
#    
#    
#    U=fft_poisson(f,h)
#    
#    
#    plt.figure()
#    #plt.imshow((U),cmap='hot')
#    plt.contour(U,50)
#    #plt.contour(f)
#    
#    plt.show()
    
    
#    display(gcf())
    
#    imshow(U)
#    
#    display(gcf())