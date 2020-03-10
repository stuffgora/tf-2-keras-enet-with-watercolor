__author__ = 'abraverm'

#from scipy.fftpack import dst,idst
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
import math as math
import scipy as sci
import numpy as np
from scipy import signal
#from scipy import misc
#from scipy import ndimage
from PIL import Image, ImageOps, ImageEnhance
import cv2
import fft_poisson_solver as fps


def get_oponent_filter() :
    s2 = math.sqrt(2.0)
    s6 = math.sqrt(6.0)
    a = 0.2989
    b = 0.5780
    c = 0.1140
#    d = 1/math.sqrt(3.0)
#    a = d
#    b = d
#    c = d
    flt =           [ [1/s2, -1/s2,  0   ],
                      [1/s6,  1/s6, -2/s6],
                      [  a ,    b ,  c   ]
                     ]
    return np.transpose(flt)
    

def to_opponent_np(x):
    oponent_transf = get_oponent_filter()
    return np.matmul(x, (oponent_transf))


def reverse_opponent_np(x):
    oponent_transf = get_oponent_filter()
    return np.matmul(x, np.linalg.inv((oponent_transf)))


def prmdDwn(img):
    i_shape = img.shape #img.shape.as_list()
    return cv2.resize(img, dsize=(i_shape[0]//2, i_shape[1]//2), interpolation=cv2.INTER_CUBIC)

def gen_img_pyramid(img,lvls):
    prmd = [img]
    for idx in range(lvls-1):
        prmd.append(prmdDwn(prmd[idx]))
    return prmd


def img_3d_conv2d(img,kernel):
    res = np.zeros(img.shape)
    for c in range (img.shape[2]):
        res[:,:,c] = signal.convolve2d(img[:,:,c], kernel, boundary='symm', mode='same')
    #return signal.convolve2d(img, kernel, boundary='symm', mode='same')
    #return ndimage.convolve(img, kernel, mode='constant', cval=0.0)
    return res

def gabor_odd_filter(img):
    #kernel = np.array([[-1,1]])
    #return img_3d_conv2d(img,kernel)
    return imgx(img)
    
def imgx(img):
    gx = np.zeros(img.shape) 
    #gx[j+1,k] = img[j+1,k] - gx[j,k]
    gx[1:,:,...] = img[1:,:,...] - img[:-1,:,...]
    return gx


def gabor_even_filter(img):
    kernel = np.array([[-1, 2, -1]])
    return img_3d_conv2d(img,kernel)
    #gx = np.zeros(img.shape) 
    #gx[1:-1,:,...] = 2*img[1:-1,:,...] - img[:-2,:,...] - img[2:,:,...]
    #return gx
    
def norm_0_1_rgb(a):
    #return (x-min(x))/(max(x)-min(x))
    #return (a - np.min(a))/np.ptp(a)
    return a/np.ptp(a)
    #return ((a - a.min(axis=(0,1)))/a.ptp(axis=(0,1)))
    #return (a-a.min())/a.ptp()
    #return np.clip(a,0,1)
    #return a - a.min(axis=(0,1))
    


def get_img_weight(img,levels):
    '''
    The calculation of the distorted edges
    gradient piramid for each chanel
    resiza back
    find max wait or sum waits  
    '''
    i_shape = img.shape #img.shape.as_list()
    imprmd = gen_img_pyramid(img,levels)
    iwx = []
    iwy = []
    wx = []
    wy = []
    for idx in range (levels):
        i = imprmd[idx]
        i = gabor_even_filter(i)
        i = cv2.resize(i, dsize=(i_shape[1], i_shape[0]), interpolation=cv2.INTER_CUBIC)
        #i = tf.image.resize_images(i, [tf.to_int32(i_shape[0]), tf.to_int32(i_shape[1])])
        iwx.append(i)
        
        i = np.transpose(imprmd[idx],[1,0,2])
        i = gabor_even_filter(i)
        i = np.transpose(i,[1,0,2])
        i = cv2.resize(i, dsize=(i_shape[1], i_shape[0]), interpolation=cv2.INTER_CUBIC)
        iwy.append(i)

        #tf.stack([x, y, z], axis=1)  # [[1, 2, 3], [4, 5, 6]]
        
    tmp = np.abs(np.stack(iwx, axis=3))
    #tmp = np.stack(iwx, axis=3)
    wx = np.amax(tmp,axis=3)
    wx = np.sum(wx,-1)
    #wx = np.sum(np.abs(iwx),(0,-1))
    wx = np.stack([wx,wx,wx] , axis=2)
    #wx = np.sum(wx,-1)
    
    
    #wx = np.sum(tmp,axis=3)
    tmp = np.abs(np.stack(iwy, axis=3))
    #tmp = np.stack(iwy, axis=3)
    wy = tf.reduce_max(tmp,axis=3)
    #wy = np.amax(tmp,axis=3)
    #wy = np.sum(tmp,axis=3)
    #tf.image.resize_images(img, tf.to_int32(i_shape[0]*2), tf.to_int32(i_shape[1]*2))
    
    wy = np.sum(wy,-1)
    #wy = np.sum(np.abs(iwy),(0,-1))
    wy = np.stack([wy,wy,wy] , axis=2)

    #return wx/np.max(wx), wy/np.max(wy)
    return wx, wy





def my_img_show(img,title):
    #pass
    #return ""
    plt.figure()
    plt.imshow(img)
    plt.title(title)
    plt.show()
    

def to_water_collor(img):
    im_pad = 3 
    i_shape = np.array(img).shape
    im_w = i_shape[0]
    im_h = i_shape[1]
    im_ch = i_shape[2]
    
    resized_image = np.zeros((im_w+2*im_pad,im_h+2*im_pad,im_ch))
    
    #padding = (im_pad,im_pad,im_pad,im_pad)
    #resized_image = ImageOps.expand(img, padding)
    #resized_image = np.array(resized_image)/256.0
    resized_image[im_pad:-im_pad,im_pad:-im_pad,:] = img
    
    my_img_opnt = to_opponent_np(resized_image)
    my_img_show(my_img_opnt,"oponent input")

    im_dx = gabor_odd_filter(my_img_opnt)

    im_dy = np.transpose(my_img_opnt,[1,0,2])
    im_dy = gabor_odd_filter(im_dy)
    im_dy = np.transpose(im_dy,[1,0,2])

    im_waits = get_img_weight(img=my_img_opnt,levels=8)
    
    alpha = 0.7
    beta = 0.4
    # element wise
    im_dx_w = np.multiply(im_dx,im_waits[0])
    im_dy_w = np.multiply(im_dy,im_waits[1])
    
    trig_xx = gabor_odd_filter(alpha*im_dx + beta*im_dx_w)
    
    trig_yy = np.transpose(alpha*im_dy + beta*im_dy_w, [1,0,2])
    trig_yy = gabor_odd_filter(trig_yy)
    trig_yy = np.transpose(trig_yy,[1,0,2])
    #alpha*(dx+dy) + beta*W*(dx+dy)
    div_trig = trig_xx + trig_yy
    
    img = div_trig
    h = 1
    res = np.zeros(img.shape)
    for c in range (img.shape[2]):
        res[:,:,c] = fps.fft_poisson(img[:,:,c], h)
    my_img_show(res,"oponent result")
    #res = norm_0_1_rgb(res)
    rgb_res = reverse_opponent_np(res)
    rgb_res = rgb_res[im_pad:-im_pad,im_pad:-im_pad,:]
    return np.clip(rgb_res,0,1)
    #return norm_0_1_rgb(rgb_res)
    #return rgb_res
    

if __name__ == '__main__':
#%%    
    im_w = 256 # 64
    im_h = 256 # 64
    im_pad = 3


    
    #img = Image.open('gray_square.png')
    #img = Image.open('stars.png')
    img = Image.open('../stars_clean.png')
    #img = Image.open('wc_ddrm.png')
    #img = Image.open('wc_blue_black.png')
    title = 'orginal image'
    my_img_show(img,title)
   
    
    im1 = img.resize((im_w, im_h), Image.ANTIALIAS)
    #padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(elta_h//2))
    padding = (im_pad,im_pad,im_pad,im_pad)
    resized_image = ImageOps.expand(im1, padding)
    resized_image = resized_image.convert("RGB")
    resized_image = np.array(resized_image)/256.0
    title = 'resized_image with padding'
    my_img_show(resized_image,title)
    
    my_img_opnt = to_opponent_np(resized_image)
    title ='my_img_opnt'
    my_img_show(my_img_opnt,title)


#    oponent_revert = reverse_opponent_np(my_img_opnt)
#    title = 'oponent reverse'
#    my_img_show(oponent_revert,title)

    y = my_img_opnt
    #title ='y'
    #show_tensor_img(y,title)

    im_dx = gabor_odd_filter(y)
    title = 'add gabor x1'
    my_img_show(im_dx,title)

    im_dy = np.transpose(y,[1,0,2])
    im_dy = gabor_odd_filter(im_dy)
    im_dy = np.transpose(im_dy,[1,0,2])
    my_img_show(im_dy,"im_dy")
#%%

    im_waits = get_img_weight(img=y,levels=4)
    #print_tensor_res(res)
    title = 'waits x'
    my_img_show(im_waits[0],title)
    title = 'waits y'
    my_img_show(im_waits[1],title)
    #show_tensor_img(res[0] + res[1])
    #show_tensor_img(res[1])
#%%
    alpha = 0.6
    beta = 0.3
    h = 1.099
    # element wise
    im_dx_w = np.multiply(im_dx,im_waits[0])
    im_dy_w = np.multiply(im_dy,im_waits[1])
    
    trig_xx = gabor_odd_filter(alpha*im_dx + beta*im_dx_w)
    
    trig_yy = np.transpose(alpha*im_dy + beta*im_dy_w, [1,0,2])
    trig_yy = gabor_odd_filter(trig_yy)
    trig_yy = np.transpose(trig_yy,[1,0,2])
    
    div_trig = trig_xx + trig_yy
    title = 'alpha(dx+dy) + beta*W*(dx+dy)'
    my_img_show(div_trig, title)

    
    img = div_trig
    
    res = np.zeros(img.shape)
    for c in range (img.shape[2]):
        res[:,:,c] = fps.fft_poisson(img[:,:,c], h)
    #return signal.convolve2d(img, kernel, boundary='symm', mode='same')
    #return ndimage.convolve(img, kernel, mode='constant', cval=0.0)
    title = 'laplace inv Mat on rerceived_img'
    my_img_show(res, title)
    
    for i in range(3):
        my_img_show(res[:,:,i], f"res channel {i}" )
        #plt.colorbar()
    
    out_img = reverse_opponent_np(res)
    title = 'RGB of rerceived_img '
    my_img_show(out_img, title)
    
    title = 'RGB norm on rerceived_img '
    my_img_show(norm_0_1_rgb(out_img), title)
    
    title = 'RGB original '
    my_img_show(im1, title)
    
#    im = (res - np.min(res))/np.ptp(res)
#    im = Image.fromarray(np.uint8(im*255))
#    #m = Image.fromarray(np.uint8(res))
#    enhancer = ImageEnhance.Brightness(im)
#    enhanced_im = enhancer.enhance(1.8)
#    my_img_show(enhanced_im, "enhanced")
