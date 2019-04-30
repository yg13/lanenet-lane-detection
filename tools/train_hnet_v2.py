#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 04/23/2019
# @Author  : Yuliang Guo
# @File    : train_hnet_v2.py
# @IDE: PyCharm Community Edition
"""
Train hnet model
"""
import argparse
import math
import os
import os.path as ops
import time
import glob

import cv2
import glog as log
import numpy as np
import tensorflow as tf

from config import global_config
from lanenet_model import hnet_model_v2
from data_provider import lanenet_hnet_data_processor

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', type=str, help='The training dataset dir path')
    parser.add_argument('--weights_path', type=str, help='The pretrained weights path')
    return parser.parse_args()


def train_net(dataset_dir, weights_path=None):
    """

    :param dataset_dir:
    :param weights_path:
    :return:
    """
    train_file_list = glob.glob('{:s}/label*.json'.format(dataset_dir))
    train_file_list = [tmp for tmp in train_file_list if '0531' not in tmp]
    val_file_list = glob.glob('{:s}/label_data_0531.json'.format(dataset_dir))

    train_dataset = lanenet_hnet_data_processor.DataSet(train_file_list)
    val_dataset = lanenet_hnet_data_processor.DataSet(val_file_list)

    with tf.device('/gpu:0'):
        input_tensor = tf.placeholder(dtype=tf.float32,
                                      shape=[CFG.TRAIN.BATCH_SIZE_HNET,
                                             CFG.TRAIN.IMG_HEIGHT_HNET,
                                             CFG.TRAIN.IMG_WIDTH_HNET, 3],
                                      name='input_tensor')
        gt_label_pts = tf.placeholder(dtype=tf.float32,
                                      shape=[CFG.TRAIN.BATCH_SIZE_HNET,
                                             CFG.DATASET.MAX_NUM_LANE,
                                             CFG.DATASET.MAX_NUM_LANE_SAMPLE, 3])

        net = hnet_model_v2.LaneNetHNet(phase=tf.constant('train', tf.string))

        # calculate the loss
        loss, coefs, losses, counter = net.compute_loss(input_tensor, gt_label_pts=gt_label_pts, name='hnet')

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(CFG.TRAIN.LEARNING_RATE_HNET, global_step,
                                                   100000, 0.1, staircase=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
            gvs = optimizer.compute_gradients(loss=loss, var_list=tf.trainable_variables())
            capped_gvs = [(tf.clip_by_value(grad, -CFG.TRAIN.CLIPPING_TH, CFG.TRAIN.CLIPPING_TH), var) for grad, var in gvs]
            clip_g_optimizer = optimizer.apply_gradients(capped_gvs, global_step=global_step)
            # optimizer = tf.train.MomentumOptimizer(
            #     learning_rate=learning_rate, momentum=0.9).minimize(loss=loss,
            #                                                         var_list=tf.trainable_variables(),
            #                                                         global_step=global_step)

    # Set tf saver
    saver = tf.train.Saver()
    model_save_dir = 'model/tusimple_hnet'
    if not ops.exists(model_save_dir):
        os.makedirs(model_save_dir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'tusimple_hnet_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)

    # Set tf summary
    tboard_save_path = 'tboard/tusimple_hnet/'
    if not ops.exists(tboard_save_path):
        os.makedirs(tboard_save_path)
    train_cost_scalar = tf.summary.scalar(name='train_cost', tensor=loss)
    val_cost_scalar = tf.summary.scalar(name='val_cost', tensor=loss)
    # train_accuracy_scalar = tf.summary.scalar(name='train_accuracy', tensor=accuracy)
    # val_accuracy_scalar = tf.summary.scalar(name='val_accuracy', tensor=accuracy)
    learning_rate_scalar = tf.summary.scalar(name='learning_rate', tensor=learning_rate)
    train_merge_summary_op = tf.summary.merge([train_cost_scalar,
                                               learning_rate_scalar])
    val_merge_summary_op = tf.summary.merge([val_cost_scalar])

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    summary_writer = tf.summary.FileWriter(tboard_save_path)
    summary_writer.add_graph(sess.graph)

    # Set the training parameters
    train_epochs = CFG.TRAIN.EPOCHS

    log.info('Global configuration is as follows:')
    log.info(CFG)

    with sess.as_default():

        tf.train.write_graph(graph_or_graph_def=sess.graph, logdir='',
                             name='{:s}/hnet_model.pb'.format(model_save_dir))

        if weights_path is None:
            log.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            log.info('Restore model from last model checkpoint {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        train_cost_time_mean = []
        val_cost_time_mean = []
        for epoch in range(train_epochs):
            # training part
            t_start = time.time()

            with tf.device('/cpu:0'):
                gt_imgs, gt_pts_list = train_dataset.next_batch(CFG.TRAIN.BATCH_SIZE_HNET)
                gt_imgs = [tmp - VGG_MEAN for tmp in gt_imgs]
            # phase_train = 'train'
            # coefs_v = sess.run(coefs, feed_dict={input_tensor: gt_imgs})
            # loss_v_np, losses_v_np, counter_v_np = hnet_model_v2.compute_loss_np(coefs_v, gt_pts_list)
            # loss_v = sess.run(loss, feed_dict={input_tensor: gt_imgs, gt_label_pts: gt_pts_list})
            _, loss_v, coefs_v, train_summary = sess.run([clip_g_optimizer, loss, coefs, train_merge_summary_op],
                                                         feed_dict={input_tensor: gt_imgs,
                                                                    gt_label_pts: gt_pts_list})
            print('pitch: \n', np.transpose(coefs_v))
            if math.isnan(loss_v):
                log.error('cost is: {:.5f}'.format(loss_v))
                cv2.imwrite('nan_image.png', gt_imgs[0] + VGG_MEAN)
                continue

            cost_time = time.time() - t_start
            train_cost_time_mean.append(cost_time)
            summary_writer.add_summary(summary=train_summary, global_step=epoch)

            # validation part
            with tf.device('/cpu:0'):
                gt_imgs_val, gt_pts_list_val = val_dataset.next_batch(CFG.TRAIN.VAL_BATCH_SIZE_HNET)
                gt_imgs_val = [tmp - VGG_MEAN for tmp in gt_imgs_val]
            # phase_val = 'test'

            t_start_val = time.time()
            # coefs_v = sess.run(coefs, feed_dict={input_tensor: gt_imgs_val})
            # loss_v_np, _, _ = hnet_model_v2.compute_loss_np(coefs_v, gt_pts_list_val)
            c_val, val_summary = sess.run([loss, val_merge_summary_op],
                                          feed_dict={input_tensor: gt_imgs_val,
                                                     gt_label_pts: gt_pts_list_val})

            summary_writer.add_summary(val_summary, global_step=epoch)

            cost_time_val = time.time() - t_start_val
            val_cost_time_mean.append(cost_time_val)

            if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                log.info('Epoch: {:d} loss= {:6f}'
                         ' mean_cost_time= {:5f}s '.
                         format(epoch + 1, loss_v,
                                np.mean(train_cost_time_mean)))
                # train_cost_time_mean.clear()

            if epoch % CFG.TRAIN.TEST_DISPLAY_STEP == 0:
                log.info('Epoch_Val: {:d} loss= {:6f} '
                         'mean_cost_time= {:5f}s '.
                         format(epoch + 1, c_val,
                                np.mean(val_cost_time_mean)))
                # val_cost_time_mean.clear()

            if epoch % 2000 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    # train hnet
    train_net(args.dataset_dir, args.weights_path)
