# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:26:30 2017

@author: shengx
"""
#%%
import UNet
from Font import *
import tensorflow as tf
import numpy as np
import time
import Parameters as params


tf.reset_default_graph()
graph = tf.Graph()
sess = tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True))
with graph.as_default():
    with tf.device("/gpu:0"):
        model = UNet.UNet()
    logger = UNet.Logger()
    init_op = tf.global_variables_initializer()

sess.run(init_op)    
logger.restore(sess)

#%%

Fonts = Font(params.basis_size, params.font_dir, params.input_letter, params.output_letter)
trainInput, trainOutput, testInput, testOutput = Fonts.getLetterSets(10510,51)
# trainInput: 10508 X 4 X 64 X 64
# trainOutput: 10508 X 1 X 64 X 64
# testInput: 51 X4 X 64 X 64
#%%
trainInput = np.swapaxes(trainInput, 1, 3)
trainOutput = np.swapaxes(trainOutput, 1, 3)
testInput = np.swapaxes(testInput, 1, 3)
testOutput = np.swapaxes(testOutput, 1, 3)

#%%
epoch = 0
begin_time = time.time()
while True:
    epoch = epoch + 1
    idx = np.random.choice(10510, params.BATCH_SIZE, replace=False)
    loss, summ = model.train(sess, trainInput[idx], trainOutput[idx])
    
    if epoch % params.VALIDATE_FREQ == 0:
        vloss, vsumm = model.train(sess, testInput[idx], testOutput[idx])
        end_time = time.time()
        training_speed = (end_time - begin_time) / params.VALIDATE_FREQ
        print('Epochs: {:d} \tVLoss: {:2.3f} \tTLoss: {:2.3f} \tSpeed: {:2.3f} sec/batch\n'.format(
                    epoch, vloss, loss, training_speed))
        begin_time = time.time()
    
    if epoch % params.SAVE_FREQ == 0:
        logger.save(sess, sess.run(model.global_step))
