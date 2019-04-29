#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 04/20/2019
# @Author  :Yuliang Guo
# @File    : hnet_v2_model.py
# @IDE: PyCharm Community Edition
"""
LaneNet中的HNet模型
"""
import tensorflow as tf
from data_provider import lanenet_hnet_data_processor
import numpy as np
import cv2

try:
    from cv2 import cv2
except ImportError:
    pass
from config import global_config
from encoder_decoder_model import cnn_basenet
from lanenet_model import hnet_v2_loss

class LaneNetHNet(cnn_basenet.CNNBaseModel):
    """
    实现lanenet中的hnet模型
    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(LaneNetHNet, self).__init__()
        self._train_phase = tf.constant('train', tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

        return

    def _init_phase(self):
        """

        :return:
        """
        return tf.equal(self._phase, self._train_phase)

    def _conv_stage(self, inputdata, out_channel, name):
        """

        :param inputdata:
        :param out_channel:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            conv = self.conv2d(inputdata=inputdata, out_channel=out_channel, kernel_size=3, use_bias=False, name='conv')
            bn = self.layerbn(inputdata=conv, is_training=self._is_training, name='bn')
            relu = self.relu(inputdata=bn, name='relu')

        return relu

    def _build_model(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            conv_stage_1 = self._conv_stage(inputdata=input_tensor, out_channel=16, name='conv_stage_1')
            conv_stage_2 = self._conv_stage(inputdata=conv_stage_1, out_channel=16, name='conv_stage_2')
            maxpool_1 = self.maxpooling(inputdata=conv_stage_2, kernel_size=2, stride=2, name='maxpool_1')
            conv_stage_3 = self._conv_stage(inputdata=maxpool_1, out_channel=32, name='conv_stage_3')
            conv_stage_4 = self._conv_stage(inputdata=conv_stage_3, out_channel=32, name='conv_stage_4')
            maxpool_2 = self.maxpooling(inputdata=conv_stage_4, kernel_size=2, stride=2, name='maxpool_2')
            conv_stage_5 = self._conv_stage(inputdata=maxpool_2, out_channel=64, name='conv_stage_5')
            conv_stage_6 = self._conv_stage(inputdata=conv_stage_5, out_channel=64, name='conv_stage_6')
            maxpool_3 = self.maxpooling(inputdata=conv_stage_6, kernel_size=2, stride=2, name='maxpool_3')
            fc = self.fullyconnect(inputdata=maxpool_3, out_dim=1024, use_bias=False, name='fc')
            fc_relu = self.relu(inputdata=fc, name='fc_relu')
            # modified network is supposed to output pitch
            output = self.fullyconnect(inputdata=fc_relu, out_dim=1, use_bias=False, name='fc_output')
            # output = tf.subtract(self.sigmoid(inputdata=output_1, name='sigmoid_out'), 0.5)
        return output

    def compute_loss(self, input_tensor, gt_label_pts, name):
        """
        计算hnet损失函数
        :param input_tensor: 原始图像[n, h, w, c]
        :param gt_label_pts: 原始图像对应的标签点集[x, y, 1]
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            trans_coefs = self._build_model(input_tensor, name=name)

            def f1():
                return [tf.constant(0.), tf.constant(0.)]

            def f2(gt_pts, valid_indices, pitch):
                gt_x = tf.gather(gt_pts[:, 0], valid_indices)
                gt_y = tf.gather(gt_pts[:, 1], valid_indices)
                gt_z = tf.gather(gt_pts[:, 2], valid_indices)
                gt_pts_new = tf.stack([gt_x, gt_y, gt_z], axis=1)

                loss = hnet_v2_loss.hnet_loss(gt_pts=gt_pts_new,
                                              pitch=pitch,
                                              name='hnet_loss')
                return [loss, tf.constant(1.)]

            # build sub-networks lane-wise, making lane-wise loss back-propagate
            losses = []
            counter = []
            # pitch = tf.multiply(tf.subtract(trans_coefs[:, 0], 0.5), np.pi)
            pitch = trans_coefs[:, 0]
            # pitch64 = tf.cast(pitch, dtype=tf.float64)
            for i in range(gt_label_pts.shape[0]):
                for j in range(gt_label_pts.shape[1]):
                    gt_pts = gt_label_pts[i, j, :, :]
                    # prune redundant padding points
                    valid_indices = tf.where(tf.greater_equal(gt_pts[:, 0], tf.zeros_like(gt_pts[:, 0])))[:, 0]
                    [loss, cnt] = tf.cond(tf.equal(tf.size(valid_indices), 0),
                                          true_fn=lambda: f1(),
                                          false_fn=lambda: f2(gt_pts, valid_indices, pitch[i]))
                    losses.append(loss)
                    counter.append(cnt)
            loss_out = tf.add(tf.divide(tf.reduce_sum(losses), tf.reduce_sum(counter)), tf.nn.l2_loss(pitch))
            # loss_out = tf.nn.l2_loss(pitch)
            return loss_out, trans_coefs, losses, counter

    def inference(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            return self._build_model(input_tensor, name=name)


def compute_loss_np(trans_coefs, gt_label_pts):
    losses = []
    counter = []
    for i, label_pts_img in enumerate(gt_label_pts):
        # pitch = (coefs_val[i, :] - 0.5) * np.pi
        pitch = trans_coefs[i, :]
        for j in range(label_pts_img.shape[0]):
            gt_pts = label_pts_img[j, :, :]
            # prune redundant padding points
            valid_indices = np.where(gt_pts[:, 0] >= 0)
            if len(valid_indices[0]) == 0:
                losses.append(0.0)
                counter.append(0.0)
                continue

            gt_pts_new = gt_pts[valid_indices[0], :]
            loss_j, _, _, _, _, _, _, _, _ = hnet_v2_loss.hnet_loss_np(gt_pts=gt_pts_new, pitch=pitch)

            losses.append(loss_j)
            counter.append(1.0)
    loss = np.sum(losses) / np.sum(counter) + np.linalg.norm(trans_coefs)
    return loss, losses, counter


if __name__ == '__main__':
    CFG = global_config.cfg
    VGG_MEAN = [103.939, 116.779, 123.68]

    input_tensor = tf.placeholder(dtype=tf.float32,
                                  shape=[CFG.TRAIN.BATCH_SIZE_HNET,
                                  CFG.TRAIN.IMG_HEIGHT_HNET,
                                  CFG.TRAIN.IMG_WIDTH_HNET, 3])
    gt_label_pts = tf.placeholder(dtype=tf.float32,
                                  shape=[CFG.TRAIN.BATCH_SIZE_HNET,
                                  CFG.DATASET.MAX_NUM_LANE,
                                  CFG.DATASET.MAX_NUM_LANE_SAMPLE, 3])

    # net1 = LaneNetHNet(phase=tf.constant('test', tf.string))
    # coefs_infer = net1.inference(input_tensor, name='hnet')
    net = LaneNetHNet(phase=tf.constant('train', tf.string))
    loss, coefs, losses, counter = net.compute_loss(input_tensor, gt_label_pts=gt_label_pts, name='hnet')

    saver = tf.train.Saver()

    train_dataset = lanenet_hnet_data_processor.DataSet(
        ['/media/yuliangguo/NewVolume2TB/Datasets/TuSimple/labeled/label_data_0313.json'])

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():
        sess.run(tf.global_variables_initializer())

        # saver.restore(sess=sess,
        #               save_path='../model/tusimple_hnet/tusimple_hnet_2019-04-17-17-58-58.ckpt-0.data-00000')
        #               # save_path='../model/tusimple_lanenet_hnet/tusimple_lanenet_hnet_2018-08-08-19-32-01.ckpt-200000')

        images, label_pts_all = train_dataset.next_batch(CFG.TRAIN.BATCH_SIZE_HNET)
        images = [tmp - VGG_MEAN for tmp in images]

        # test loss function using numpy function for comparison
        coefs_val = sess.run(coefs, feed_dict={input_tensor: images})
        print('Network output coefs: \n', coefs_val)

        loss_val, losses_val, counter_val = compute_loss_np(coefs_val, label_pts_all)
        print('\n loss results from numpy implementation')
        print('loss: {:.5f}'.format(loss_val))
        print('losses: ', losses_val)
        print('counter:', counter_val)

        loss_val, losses_val, counter_val = sess.run([loss, losses, counter],
                                                     feed_dict={input_tensor: images,
                                                     gt_label_pts: label_pts_all})
        print('\n loss results computed from TF implementation (float64)')
        print('loss: {:.5f}'.format(loss_val))
        print('losses: ', losses_val)
        print('counter:', counter_val)
        # print('label_pts_all: ', label_pts_all)

        # print the output of homograph net
        print('\n Examine the first homography matrix')
        Tz = CFG.DATASET.Tz
        f = CFG.DATASET.f
        h_org = CFG.DATASET.IMG_HEIGHT_ORG
        w_org = CFG.DATASET.IMG_WIDTH_ORG
        h_hnet = CFG.TRAIN.IMG_HEIGHT_HNET
        w_hnet = CFG.TRAIN.IMG_WIDTH_HNET
        fx = f * float(w_hnet) / float(w_org)
        fy = f * float(h_hnet) / float(h_org)
        cx = float(w_hnet) * 0.5
        cy = float(h_hnet) * 0.5
        # pitch = (coefs_val[0, 0] - 0.5) * np.pi
        pitch = coefs_val[0, 0]
        c1 = np.cos(pitch)
        s1 = np.sin(pitch)
        K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
        print('K: \n', K)
        Ex = np.array([[1., 0., 0.], [0., s1, c1*Tz], [0., c1, s1*Tz]])
        print('Ex: \n', Ex)
        H_inv = np.matmul(K, Ex)
        print('H_inv: \n', H_inv)
        H = np.mat(H_inv).I
        print('H: \n', H)

        # warp_image = cv2.warpPerspective(images[0], H, dsize=(images[0].shape[1], images[0].shape[0]))
        # cv2.imwrite("src.jpg", images[0] + VGG_MEAN)
        # cv2.imwrite("ret.jpg", warp_image)
