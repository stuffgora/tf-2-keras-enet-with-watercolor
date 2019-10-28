#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 21:03:33 2019

@author: ddd
"""

import argparse
#from argparse import Namespace

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class EnetCfg():
    
    def __init__(self,json_f=None,is_cli_arg=1): # **kwargs):
        self.parser = argparse.ArgumentParser()
        self.json_f = json_f
        self.is_cli_arg = is_cli_arg
        super(EnetCfg, self).__init__() #**kwargs)
        
    def add_attr(self,var,value,desc="",is_str=None):
        if is_str:
            exec(f'self.{var}="{value}"')
        else:
            exec(f'self.{var}={value}')
        
    def DEFINE_string(self,var,value,desc=""):
        if self.is_cli_arg:
            self.parser.add_argument(f"--{var}", type=str, default=value, help=desc)
        self.add_attr(var,value,desc,is_str=1)
    
    def DEFINE_integer(self,var,value,desc=""):
        if self.is_cli_arg:
            self.parser.add_argument(f"--{var}", type=int, default=value, help=desc)
        self.add_attr(var,value,desc)
    
    def DEFINE_float(self,var,value,desc=""):
        if self.is_cli_arg:
            self.parser.add_argument(f"--{var}", type=float, default=value, help=desc)
        self.add_attr(var,value,desc)
    
    def DEFINE_boolean(self,var,value,desc=""):
        if self.is_cli_arg:
            self.parser.add_argument(f"--{var}", type=bool, default=value, help=desc)
        self.add_attr(var,value,desc)
    
    def  parse_args(self):
        r = self.parser.parse_args()
        #self.print_args()
        return r 
    
    def default_enet_cfg(self,flags=None):
        self.DEFINE_string('dataset_name', 'camvid', 'Data Set neme camvid/coco.')
        self.DEFINE_string('model_name', 'enet_x', 'The dataset directory to find the train, validation and test images.')
        self.DEFINE_string('dataset_dir', './dataset', 'The dataset directory to find the train, validation and test images.')
        self.DEFINE_integer('num_classes', 12, 'The number of classes to predict.')
        self.DEFINE_integer('batch_size', 10, 'The batch_size for Tes.')
        self.DEFINE_integer('val_batch_size', 25, 'The batch_size for Tes.')
        self.DEFINE_integer('concat_ds', 0, 'If > 0 concatinates val and train for training.')
        self.DEFINE_integer('repeat_train_ds', None, 'If n > 0  repeat n time, if 0 repeat inf, if None do not repeat train datat set.')

        self.DEFINE_integer('steps', None, 'Number of batches in epoch')
        self.DEFINE_integer('epochs', 100, 'Number of epochs to proceed')
        self.DEFINE_integer('image_height', 360, "The input height of the images.")
        self.DEFINE_integer('image_width', 480, "The input width of the images.")
        self.DEFINE_integer('val_steps', None, 'Number of val batches to proceed each epoch')
        self.DEFINE_integer('initial_epoch', None, 'Number of already compleated epochs')
        
        #self.DEFINE_string('weights_file', 'models/wc_camvid/enet/weights/enet_best.h5', 'Path to pretrained model.')
        #self.DEFINE_string('train_model', 'wc_camvid', 'Data Set neme.')
        self.DEFINE_integer('wc_in_encoder', None, "None no wc in encoder, 1 wc on first 3 conv chanels, 2 wc on max_pull, 3 wc on input added to all 16 channels ")
        self.DEFINE_integer('wc_in_decoder', None, "None - No wc in decider, 1 - wc before bptelneck 5 with upsample")
        self.DEFINE_integer('wc_lpl_mat', None, "None - use fft, othet value - use laplace inv matrix in water collor layer")
        self.DEFINE_string('wc_activation', None , "None - use no activation, <actv> - use <actv> in water collor midle layers, for example 'tanh/relu'" )

        self.DEFINE_integer('train_flow', 1, "train model")
        self.DEFINE_integer('predict_flow', 0, "predict 5 example samples from test data")
        self.DEFINE_integer('eval_flow', 0, "evaluete test datatset")
        

        
        #self.parse_args()

# # Example Mutually Exclusive_group
#        group = parser.add_mutually_exclusive_group()
#        group.add_argument("-v", "--verbose", action="store_true")
#        group.add_argument("-q", "--quiet", action="store_true")

    def print_args(self):
        cfg = self.parser.parse_args()
        print("\nCurent ENet_WC configuration." )
        print("-----------")
        #print(str(cfg).split('(')[1].split(')')[0].split(','))          
        #print(cfg)
        arg_col_width = max(len(arg) for arg in vars(cfg)) + 2
        for arg in vars(cfg):
            print (arg.ljust(arg_col_width),":", getattr(cfg, arg))
        print("-----------\n")
        
if __name__ == '__main__':
    flags = EnetCfg()
    flags.DEFINE_string('dataset_dir', './dataset', 'The dataset directory to find the train, validation and test images.')
    flags.DEFINE_integer('num_classes', 12, 'The number of classes to predict.')
    flags.DEFINE_integer('batch_size', 10, 'The batch_size for training.')
    flags.DEFINE_boolean('skip_connections', False, 'If True, perform skip connections from encoder to decoder.')
    #redefine
    flags.DEFINE_integer('num_classes', 13, 'The number of classes to predict.')
    FLAGS = flags        
    
    flags_cli = EnetCfg(is_cli_arg=1)      
    flags_cli.DEFINE_string('dataset_dir', './dataset', 'The dataset directory to find the train, validation and test images.')
    flags_cli.DEFINE_integer('num_classes', 12, 'The number of classes to predict.')
    flags_cli.DEFINE_integer('batch_size', 10, 'The batch_size for training.')
    flags_cli.DEFINE_boolean('skip_connections', False, 'If True, perform skip connections from encoder to decoder.')
    FLAGS_cli =  flags_cli.parse_args()
     
 

    print (f"{FLAGS.dataset_dir, FLAGS.num_classes, FLAGS.batch_size, FLAGS.skip_connections, }")        
    
    print(flags_cli)
    print(FLAGS_cli.dataset_dir,FLAGS_cli.num_classes,FLAGS_cli.batch_size,FLAGS_cli.skip_connections)
    print(FLAGS_cli)
    #redefine
    FLAGS_cli.num_classes =  13
    print(FLAGS_cli)
        
#{
# "model_name": "enet",
# "nb_epoch": 100,
# "batch_size": 8,
# "completed_epochs": 0,
# "dh": 256,
# "dw": 256,
# "skip": 0,
# "h5file": "models/camvid/enet/weights/enet_best.h5"
#}