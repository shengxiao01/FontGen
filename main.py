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
from matplotlib import pyplot as plt

Fonts = Font()
trainInput, trainOutput, testInput, testOutput = Fonts.getLetterSets()

#%%
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
epoch = 0
begin_time = time.time()
while True:
    epoch = epoch + 1
    idx = np.random.choice(len(trainInput), params.BATCH_SIZE, replace=False)
    loss, summ = model.train(sess, trainInput[idx], trainOutput[idx])
    
    if epoch % params.VALIDATE_FREQ == 0:
        test_idx = np.random.choice(len(testInput), params.BATCH_SIZE, replace=False)
        vloss, vsumm = model.test(sess, testInput[test_idx], testOutput[test_idx])
        logger.save_summary(epoch, train_summary = summ, test_summary = vsumm)
        end_time = time.time()
        training_speed = (end_time - begin_time) / params.VALIDATE_FREQ
        print('Epochs: {:d} \tVLoss: {:2.3f} \tTLoss: {:2.3f} \tSpeed: {:2.3f} sec/batch'.format(
                    epoch, vloss, loss, training_speed))
        begin_time = time.time()
    if epoch % 60 == 0:
        predicted_img = model.predict(sess, testInput[test_idx])
        plt.figure(1)
        plt.axis('off')
        plt.subplot(281)
        plt.imshow(predicted_img[1, :, :, 0],interpolation="nearest",cmap='Greys')
        plt.subplot(282)
        plt.imshow(predicted_img[1, :, :, 1],interpolation="nearest",cmap='Greys')
        plt.subplot(283)
        plt.imshow(predicted_img[1, :, :, 2],interpolation="nearest",cmap='Greys')
        plt.subplot(284)
        plt.imshow(predicted_img[1, :, :, 3],interpolation="nearest",cmap='Greys')    
        plt.subplot(285)
        plt.imshow(predicted_img[1, :, :, 4],interpolation="nearest",cmap='Greys')
        plt.subplot(286)
        plt.imshow(predicted_img[1, :, :, 5],interpolation="nearest",cmap='Greys')
        plt.subplot(287)
        plt.imshow(predicted_img[1, :, :, 6],interpolation="nearest",cmap='Greys')
        plt.subplot(288)
        plt.imshow(predicted_img[1, :, :, 7],interpolation="nearest",cmap='Greys')           
        plt.subplot(289)
        plt.imshow(testOutput[test_idx[1], :, :, 0],interpolation="nearest",cmap='Greys')
        plt.show()
        plt.pause(0.0001) 
    if epoch % params.SAVE_FREQ == 0:
        logger.save(sess, sess.run(model.global_step))
#%%
predicted_img = model.predict(sess, testInput)
