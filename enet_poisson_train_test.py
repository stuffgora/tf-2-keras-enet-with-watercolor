#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 01:34:41 2020

@author: ddd
"""

#import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from src.enet_wc_keras_model import autoencoder_wc
from src.enetcfg import EnetCfg

from PIL import Image
from src.watercolor02_np_fft import to_water_collor, my_img_show

flags = EnetCfg()
flags.default_enet_cfg(flags=flags)
cfg = flags.parse_args()

def rand_img(m,n):
#    img = '../stars_clean.png'    
#    x = np.array(Image.open(img).convert("RGB"))/256.0 
    for i in range(1000):
        x = np.random.rand(m,n,3)
        y = to_water_collor(x)
        yield (x,y)
    
    
def get_model(cfg=cfg):
    dw = cfg.image_width
    dh = cfg.image_height
    
    optimizer = 'adam' #get_optimizer()
    loss = 'mean_squared_error' #get_loss()
    metrics=['accuracy', 'mean_squared_error'] #get_metrics()
    
    model, model_name = autoencoder_wc(nc=3, input_shape=(dw, dh),
                                         loss=loss,
                                         optimizer=optimizer, 
                                         metrics=metrics, 
                                         #wc_in_encoder=cfg.wc_in_encoder,
                                         #wc_in_decoder=cfg.wc_in_decoder
                                         )
    return model

def train( ):
    print(f'Preparing to train on {cfg.dataset_name} data...')
    
    autoenc = get_model()
    #autoenc, ckpt = model_preload(autoenc) 
    #train_ds, val_ds = get_train_val_data(dataset = cfg.dataset_name)
    
    # Class Waight balancing        
    #cw_d = get_class_normalization_dic(dataset=cfg.dataset_name)
    #checkpoint_dir, log_dir = get_checkpoint_log_dir()
    # checkpoint for epoch counter
#    if cfg.initial_epoch is None:
#        completed_epochs = ckpt.completed_epochs.numpy()
#    else :
#        completed_epochs = cfg.initial_epoch  
    m = cfg.image_width
    n = cfg.image_height
    
    train_ds = tf.data.Dataset.from_generator(rand_img, 
                                              output_types=(tf.float32, tf.float32),
                                              output_shapes=(tf.TensorShape([m,n,3]), tf.TensorShape([m,n,3])),
                                              args=(m,n)
                                              )
    val_ds = tf.data.Dataset.from_generator(rand_img, 
                                              output_types=(tf.float32, tf.float32),
                                              output_shapes=(tf.TensorShape([m,n,3]), tf.TensorShape([m,n,3])),
                                              args=(m,n)
                                              )
    # ---------- Fit Model - Training-----------------
    history = autoenc.fit(
        x=train_ds.batch(100),
        epochs=cfg.epochs,
        steps_per_epoch=cfg.steps,
        #class_weight=cw_d,
        verbose=1,
        #callbacks=callbacks(log_dir, checkpoint_dir, cfg.model_name ),        
        validation_data=val_ds.batch(10),
        validation_steps=cfg.val_steps,
        #initial_epoch=completed_epochs
    )  
    print('\nhistory dict:', history.history)
    return autoenc


if __name__ == '__main__':
    m = cfg.image_width
    n = cfg.image_height
    
    img = 'stars_clean.png'    
    x = np.array(Image.open(img).resize((m, n)).convert("RGB"))/256.0
    
    model = get_model()
    model.save('saved_model/poisson_model')
    
    model = train( )
    model.save('saved_model/poisson_model')
    #new_model = tf.keras.models.load_model('saved_model/my_model')
    
    
    
    
    x = x.reshape((1,m,n,3))
    y = model.predict(x)
    y = x.reshape((m,n,3))
    my_img_show(y, "res")
    