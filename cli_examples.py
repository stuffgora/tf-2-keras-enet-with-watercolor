#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Train  loop
#for i in {0..50}
# do
#  let eps=20+$i*20
#  let i_ep=$i*20
#  python enet_keras_wc_train_val_pred.py --dataset_name camvid --model_name enet_wc_before_encoder_256x256 --image_height 256 --image_width 256 --num_classes 12 --batch_size 10 --val_batch_size 20 --repeat_train_ds 0 --steps 100 --wc_in_encoder 0 --train_flow 1 --predict_flow 0 --eval_flow 0 --epochs "$eps" --initial_epoch "$i_ep"
# done


# Predict {batch_size} images from test dataset with enet_wc_before_encoder_256x256
python enet_keras_wc_train_val_pred.py --dataset_name camvid --model_name enet_wc_before_encoder_256x256 --image_height 256 --image_width 256 --num_classes 12 --batch_size 10 --val_batch_size 20 --repeat_train_ds 0 --steps 100 --wc_in_encoder 0 --train_flow 0 --predict_flow 1 --eval_flow 0 --epochs 50

# Predict {batch_size} images from test dataset with enet_no_wc_256x256
python enet_keras_wc_train_val_pred.py --dataset_name camvid --model_name enet_no_wc_256x256 --image_height 256 --image_width 256 --num_classes 12 --batch_size 10 --val_batch_size 20 --repeat_train_ds 0 --steps 100 --train_flow 0 --predict_flow 1 --eval_flow 0 --epochs 50

# Predict {batch_size} images from test dataset with enet_wc_in_encoder_1_256x256
python enet_keras_wc_train_val_pred.py --dataset_name camvid --model_name enet_wc_in_encoder_1_256x256 --image_height 256 --image_width 256 --num_classes 12 --batch_size 10 --val_batch_size 20 --repeat_train_ds 0 --steps 100 --wc_in_encoder 1 --train_flow 0 --predict_flow 1 --eval_flow 0 --epochs 50
