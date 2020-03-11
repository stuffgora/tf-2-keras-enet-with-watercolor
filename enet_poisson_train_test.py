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



gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

flags = EnetCfg()
flags.default_enet_cfg(flags=flags)
cfg = flags.parse_args()


def rand_img(m,n):
#    img = '../stars_clean.png'    
#    x = np.array(Image.open(img).convert("RGB"))/256.0 
    #for i in range(100000):
    while True:
        x = np.random.rand(m,n,3)
        y = to_water_collor(x)
        yield (x,y)

def get_rand_data(m,n):
    return tf.data.Dataset.from_generator(
            rand_img, 
            output_types=(tf.float32, tf.float32),
            output_shapes=(tf.TensorShape([m,n,3]), tf.TensorShape([m,n,3])),
            args=(m,n)
            )


def get_optimizer():
    initial_learning_rate = 5e-4 #0.1
    #decay_steps = int(num_epochs_before_decay * num_steps_per_epoch) ~100*100
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=500, #10*2*(50*10),#    (steps_in_s*batch)
        decay_rate=1e-1, #0.96,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)
    #optimizer = tf.optimizers.Adadelta(learning_rate=5e-5, rho=1e-1, epsilon=1e-08)
    #optimizer='adadelta'
    return optimizer
   
def get_model(cfg=cfg):
    dw = cfg.image_width
    dh = cfg.image_height
    
    optimizer = get_optimizer() #'SGD' # 'adam' #
    loss = 'mean_squared_error' #get_loss()
    metrics=['accuracy', 'mean_squared_error'] #get_metrics()
    
    model, model_name = autoencoder_wc(nc=3, 
                                       input_shape=(dw, dh),
                                       loss=loss,
                                       optimizer=optimizer, 
                                       metrics=metrics )                                      
    return model


def train( ):
    print(f'Preparing to train on {cfg.dataset_name} data...')
    
    autoenc = get_model()
    #model.save('saved_model/poisson_model')
    pre_trained_model = tf.keras.models.load_model('saved_model/poisson_model')
    
    autoenc.set_weights(pre_trained_model.get_weights())
 
    m = cfg.image_width
    n = cfg.image_height
    
    train_ds = get_rand_data(m,n)
    val_ds   = get_rand_data(m,n)
    
    # ---------- Fit Model - Training-----------------
    history = autoenc.fit(
        x=train_ds.batch(5),
        epochs=100, #200, #cfg.epochs,
        steps_per_epoch=10, #cfg.steps,
        #class_weight=cw_d,
        verbose=1,
        #callbacks=callbacks(log_dir, checkpoint_dir, cfg.model_name ),        
        validation_data=val_ds.batch(2),
        validation_steps=4 #cfg.val_steps,
        #initial_epoch=completed_epochs
    )  
    print('\nhistory dict:', history.history)
    return autoenc


if __name__ == '__main__':
    cfg.image_width  = 256
    cfg.image_height = 256
    m = cfg.image_width
    n = cfg.image_height
    
    img = 'stars_clean.png'    
    x = np.array(Image.open(img).resize((n, m)).convert("RGB"))/256.0
    
    #model = get_model()
    #model.save('saved_model/poisson_model')
    
    model = train( )
    model.save('saved_model/poisson_model')
    #new_model = tf.keras.models.load_model('saved_model/my_model') 
       
    x = np.expand_dims(x, axis=0)
    y = model.predict(x)
    y = np.squeeze(y, axis=0)
    my_img_show(y, "res")
 
    
    
    
    
    
    
    
    
    
    
    
    
    