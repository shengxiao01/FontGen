# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:28:30 2017

@author: shengx
"""

SAVE_PATH = './save/'

font_dir = '../10000_Fonts'
input_letter = ['A','B','S','Q', 'X']
output_letter = ['E', 'G', 'R', 'W', 'Z']


BATCH_SIZE = 24
basis_size = 96

IMG_X = basis_size
IMG_Y = basis_size
IMG_Z = len(input_letter)
IMG_OUT_Z = len(output_letter)

LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.99
LEARNING_RATE_DECAY_STEP = 1000
WEIGHT_L2_REG = 0.0002

VALIDATE_FREQ = 20
SAVE_FREQ = 1000
TRAIN_TEST_SPLIT = 20

DROPOUT_RATE = 0.1