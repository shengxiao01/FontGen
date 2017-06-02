# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:26:30 2017

@author: shengx
"""
#%%
import UNet
import Font 
import tensorflow as tf
import numpy as np
import time
import Parameters as params

Fonts = Font.Font()
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
epoch = sess.run(model.global_step)
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
        test_display(predicted_img, testOutput[test_idx], 0)
    if epoch % params.SAVE_FREQ == 0:
        logger.save(sess, sess.run(model.global_step))
#%%
