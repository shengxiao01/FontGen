# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:28:30 2017

@author: shengx
"""

SAVE_PATH = './save/'

font_dir = '../10000_Fonts'
input_letter = ['B','A','S','Q']
output_letter = ['R']


BATCH_SIZE = 128
basis_size = 64

IMG_X = basis_size
IMG_Y = basis_size
IMG_Z = len(input_letter)

LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.99
LEARNING_RATE_DECAY_STEP = 1000
WEIGHT_L2_REG = 0.0002

VALIDATE_FREQ = 20
SAVE_FREQ = 1000