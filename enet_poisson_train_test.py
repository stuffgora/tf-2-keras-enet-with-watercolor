#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 01:34:41 2020

@author: ddd
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from src.enet_wc_keras_model import autoencoder_wc
from src.enetcfg import EnetCfg


flags = EnetCfg()
flags.default_enet_cfg(flags=flags)
cfg = flags.parse_args()


def get_model(cfg=cfg):
    dw = cfg.image_width
    dh = cfg.image_height
    
    optimizer = 'adam' #get_optimizer()
    loss = 'mean_squared_error' #get_loss()
    metrics=['accuracy', 'mean_squared_error'] #get_metrics()
    
    model, model_name = autoencoder_wc(nc=cfg.num_classes, input_shape=(dw, dh),
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
    autoenc, ckpt = model_preload(autoenc) 
    train_ds, val_ds = get_train_val_data(dataset = cfg.dataset_name)
    
    # Class Waight balancing        
    cw_d = get_class_normalization_dic(dataset=cfg.dataset_name)
    checkpoint_dir, log_dir = get_checkpoint_log_dir()
    # checkpoint for epoch counter
    if cfg.initial_epoch is None:
        completed_epochs = ckpt.completed_epochs.numpy()
    else :
        completed_epochs = cfg.initial_epoch  
    # ---------- Fit Model - Training-----------------
    history = autoenc.fit(
        x=train_ds,
        epochs=cfg.epochs,
        steps_per_epoch=cfg.steps,
        class_weight=cw_d,
        verbose=1,
        callbacks=callbacks(log_dir, checkpoint_dir, cfg.model_name ),        
        validation_data=val_ds,
        validation_steps=cfg.val_steps,
        initial_epoch=completed_epochs
    )  
    print('\nhistory dict:', history.history)
    return autoenc