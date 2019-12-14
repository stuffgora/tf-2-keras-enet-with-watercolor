# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 00:24:54 2019

@author: abraverm
"""

import tensorflow as tf
from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt
import numpy as np
#import tkinter

from wc_keras_model import get_wc_model 
from enetcfg import EnetCfg
from enet_wc_keras_model import WcLayer, wc_zero_layer
from wc_keras_model import  decode_img, my_show_image, show_tensor_img,get_oponent_filter

import sys
sys.path.append('../')
from enet_keras_wc_train_val_pred import get_model



def to_opponent_np(x):
    oponent_transf = get_oponent_filter()
    return np.matmul(x, (oponent_transf))


def reverse_opponent_np(x):
    oponent_transf = get_oponent_filter()
    return np.matmul(x, np.linalg.inv((oponent_transf)))



if __name__ == '__main__':
    
    flags = EnetCfg()
    flags.default_enet_cfg()
    cfg = flags.parse_args()
    cfg.image_width = 256
    cfg.image_height = 256
    cfg.wc_in_encoder = 0
    cfg.wc_in_decoder = None   
    #cfg = flags.parse_args(['--wc_in_encoder', '0', '--image_width', '256', '--image_height', '256'])
    im_h = cfg.image_height
    im_w = cfg.image_width
    im_c = 3
    
    img_file = '../gray_square.png'
    img_file = '../dataset/test/Seq05VD_f04770.png'
    img = tf.io.read_file(img_file)
    
    my_img_rgb = decode_img(img, im_h, im_w)

    my_show_image(my_img_rgb, "source img")
    im1 = tf.image.resize_with_pad(my_img_rgb, im_h, im_w)
    im1 = tf.reshape(im1,[1,im_h,im_w,im_c], name="input_reshaped")
    show_tensor_img(im1[0],"Source image tf")
    
    input_shape=(cfg.image_width, cfg.image_height) 
    inp =  Input(shape=(input_shape[0], input_shape[1], 3))
    model = inp
    model = wc_zero_layer(model)
    model = tf.keras.Model(inputs=inp, outputs=model)
    
    
    
    res = model.predict(im1)
    # cocatinated inp with output
    res = res[0,:,:,3:]
    #res = tf.reshape(res,[res.shape[1],res.shape[2],3])
    #show_tensor_img(res,"result image")
    my_show_image(res,"result image")
    my_show_image(res[:,:,0],"result image[0]")
    
    norm_res = res
    for i in range(3):
        tmp = res[:,:,i]
        tmp -= np.min(tmp)
        tmp = tmp/np.max(tmp)
        norm_res[:,:,i] = tmp 
    my_show_image(norm_res,"norm result image")
    rgb_norm_res = reverse_opponent_np(norm_res)
    my_show_image(rgb_norm_res,"norm RGB result image")
    
    #plt.imshow(res[:,:,0])
    #plt.show()
    #model preload
    if 1>0:
        cfg.image_width = 256
        cfg.image_height = 256
        cfg.wc_in_encoder = 0
        cfg.wc_in_decoder = None
        # first 7 layers : inp -> conv2d->maxpull->concat->[0:3] init -> conv2d->norm->prelu 
        enet_model = get_model(cfg)
        #best_weights = 'models/coco/enet_no_wc_256x256/weights/enet_no_wc_256x256_best.hdf5'
        best_weights = '../models/camvid/enet_wc_before_encoder_256x256/weights/enet_wc_before_encoder_256x256_best.hdf5'
        enet_model.load_weights(best_weights)
        print(enet_model.layers[2].get_weights())
        for idx in range(2):
            model.layers[idx].set_weights(enet_model.layers[idx].get_weights())
        
        res = model.predict(im1)
        res = res[0,:,:,3:]
        my_show_image(res,"result image preloaded")
        my_show_image(res[:,:,0],"result image preloaded[0]")
        
#        cfg.wc_in_encoder = 1
#        wc_model = get_model()
#        offset = len(wc_model.layers) - len(enet_model.layers)
#        for idx in range(7,len(enet_model.layers)):
#           wc_model.layers[idx+offset].set_weights(enet_model.layers[idx].get_weights())
#           wc_model.layers[idx+offset].trainable = False

    if 0>0:
        cfg.image_width = 256
        cfg.image_height = 256
        cfg.wc_in_encoder = 0
        cfg.wc_in_decoder = None
        model = get_model()
        print("model.summary()")
        with open('model_sum.txt','w') as f:
            print("wriet file")
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(' ---------- len(model.trainable_variables) ------ ')
        print('trainable_variables #:',len(model.trainable_variables))
        print('layers #:',len(model.layers))
        tf.keras.utils.plot_model(model, f'{cfg.model_name}_model_with_shape_info.png', show_shapes=True)
        
        
        
        