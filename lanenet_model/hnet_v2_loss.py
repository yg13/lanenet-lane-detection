#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 04/20/2019
# @Author  : Yuliang Guo
# @File    : hnet_v2_loss.py
# @IDE: PyCharm Community Edition
"""
实现LaneNet的HNet损失函数
"""
import tensorflow as tf
import numpy as np


def hnet_loss(gt_pts, pitch, name):
    """
    :param gt_pts: image points [x, y, 1]
    :param H: homograph between image and ground plane [[a, b, c], [0, d, e], [0, f, 1]]
    :param name:
    :return:
    """
    with tf.variable_scope(name):

        # modified network is supposed to output pitch angle only
        # compute homograph matrix

        c1 = tf.cos(pitch)
        s1 = tf.sin(pitch)

        K = tf.constant([100.0, 0.0, 64.0,
                         0.0, 100.0, 32.0,
                         0.0, 0.0, 1.0])
        K = tf.reshape(K, shape=[3, 3])
        Ex = tf.stack([tf.constant(1.0), tf.constant(0.0), tf.constant(0.0),
                      tf.constant(0.0), s1, tf.constant(1.5),
                      tf.constant(0.0), c1, tf.constant(0.0)])
        Ex = tf.reshape(Ex, shape=[3, 3])
        H_inv = tf.matmul(K, Ex)
        H = tf.matrix_inverse(H_inv)

        gt_pts = tf.transpose(gt_pts)
        pts_projects = tf.matmul(H, gt_pts)

        # second-order polynomial fitting
        X = tf.transpose(pts_projects[0, :])
        Y = tf.transpose(pts_projects[1, :])
        X = tf.multiply(X, tf.reciprocal(pts_projects[2, :]))
        Y = tf.multiply(Y, tf.reciprocal(pts_projects[2, :]))

        # Y_One = tf.add(tf.subtract(Y, Y), tf.constant(1.0, tf.float32))
        Y_One = tf.ones_like(Y, tf.float32)
        Y_stack = tf.stack([tf.pow(Y, 3), tf.pow(Y, 2), Y, Y_One], axis=1)
        w = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(Y_stack), Y_stack)),
                                tf.transpose(Y_stack)), tf.expand_dims(X, -1))

        # compute re-projection error in image coordinates
        x_preds = tf.matmul(Y_stack, w)
        preds = tf.transpose(tf.stack([tf.squeeze(x_preds, -1), Y, Y_One], axis=1))
        pts_trans_back = tf.matmul(H_inv, preds)
        X_trans_back = tf.multiply(pts_trans_back[0, :], tf.reciprocal(pts_trans_back[2, :]))

        loss = tf.reduce_mean(tf.pow(gt_pts[0, :] - X_trans_back, 2))

    return loss


def hnet_transformation(gt_pts, trans_coef, name):
    """
    :param gt_pts: image points [x, y, 1]
    :param H: homograph between image and ground plane [[a, b, c], [0, d, e], [0, f, 1]]
    :param name:
    :return:
    """

    with tf.variable_scope(name):
        # modified network is supposed to output trans_coef = [fx/160, fy/120, cx/160, cy/160, pitch]
        # compute homograph matrix
        fx = tf.multiply(trans_coef[0], 160)
        fy = tf.multiply(trans_coef[1], 120)
        cx = tf.multiply(trans_coef[2], 160)
        cy = tf.multiply(trans_coef[3], 120)
        c1 = tf.cos(trans_coef[4])
        s1 = tf.sin(trans_coef[4])

        # K = tf.stack([fx, tf.constant(0.0), cx,
        #               tf.constant(0.0), fy, cy,
        #               tf.constant(0.0), tf.constant(0.0), tf.constant(1.0)])
        K = tf.constant([100.0, 0.0, 64.0,
                         0.0, 100.0, 32.0,
                         0.0, 0.0, 1.0])
        K = tf.reshape(K, shape=[3, 3])
        Ex = tf.stack([tf.constant(1.0), tf.constant(0.0), tf.constant(0.0),
                      tf.constant(0.0), s1, tf.constant(1.5),
                      tf.constant(0.0), c1, tf.constant(0.0)])
        Ex = tf.reshape(Ex, shape=[3, 3])
        H_inv = tf.matmul(K, Ex)
        H = tf.matrix_inverse(H_inv)

        gt_pts = tf.transpose(gt_pts)
        pts_projects = tf.matmul(H, gt_pts)

        # second-order polynomial fitting
        X = tf.transpose(pts_projects[0, :])
        Y = tf.transpose(pts_projects[1, :])
        X = tf.multiply(X, tf.reciprocal(pts_projects[2, :]))
        Y = tf.multiply(Y, tf.reciprocal(pts_projects[2, :]))

        # Y_One = tf.add(tf.subtract(Y, Y), tf.constant(1.0, tf.float32))
        Y_One = tf.ones_like(Y, tf.float32)
        Y_stack = tf.stack([tf.pow(Y, 3), tf.pow(Y, 2), Y, Y_One], axis=1)
        w = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(Y_stack), Y_stack)),
                                tf.transpose(Y_stack)), tf.expand_dims(X, -1))

        # compute re-projection error in image coordinates
        x_preds = tf.matmul(Y_stack, w)
        preds = tf.transpose(tf.stack([tf.squeeze(x_preds, -1), Y, Y_One], axis=1))
        pts_trans_back = tf.matmul(H_inv, preds)
        X_trans_back = tf.multiply(pts_trans_back[0, :], tf.reciprocal(pts_trans_back[2, :]))
        # X_trans_back = pts_trans_back[0, :]

    return gt_pts, X_trans_back, X, Y


# def hnet_transformation(gt_pts, H, name):
#     """
#     :param gt_pts:
#     :param H:
#     :param name:
#     :return:
#     """
#     with tf.variable_scope(name):
#
#         gt_pts = tf.transpose(gt_pts)
#         pts_projects = tf.matmul(H, gt_pts)
#
#         # second-order polynomial fitting
#         Y = tf.transpose(pts_projects[1, :])
#         X = tf.transpose(pts_projects[0, :])
#         Y_One = tf.add(tf.subtract(Y, Y), tf.constant(1.0, tf.float32))
#         Y_stack = tf.stack([tf.pow(Y, 3), tf.pow(Y, 2), Y, Y_One], axis=1)
#         w = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(Y_stack), Y_stack)),
#                                 tf.transpose(Y_stack)),
#                       tf.expand_dims(X, -1))
#
#         # compute re-projection error in image coordinates
#         x_preds = tf.matmul(Y_stack, w)
#         preds = tf.transpose(tf.stack([tf.squeeze(x_preds, -1), Y, Y_One], axis=1))
#         preds_fit = tf.stack([tf.squeeze(x_preds, -1), Y], axis=1)
#         x_transformation_back = tf.matmul(tf.matrix_inverse(H), preds)
#
#     return x_transformation_back


if __name__ == '__main__':
    # gt_labels = tf.constant([[[1.0, 1.0, 1.0], [2.0, 2.0, 1.0], [3.0, 3.0, 1.0]],
    #                          [[4.0, 4.0, 1.0], [5.0, 5.0, 1.0], [6.0, 6.0, 1.0]]],
    #                         dtype=tf.float32, shape=[6, 3])
    gt_labels = tf.constant([[[0.0, 60.0, 1.0], [0.0, 55.0, 1.0], [0.0, 50.0, 1.0]],
                             [[0.0, 45.0, 1.0], [0.0, 40.0, 1.0], [0.0, 35.0, 1.0]]],
                            dtype=tf.float32, shape=[6, 3])
    pitch = tf.contant(0.0)
    Tz = 1.5
    c1 = np.cos(pitch)
    s1 = np.sin(pitch)

    K = np.array([100.0, 0.0, 64.0,
                  0.0, 100.0, 32.0,
                  0.0, 0.0, 1.0])
    K = np.reshape(K, [3, 3])
    print('K: ', K)
    Ex = np.array([1.0, 0.0, 0.0,
                   0.0, s1, c1*Tz,
                   0.0, c1, s1*Tz])
    Ex = np.reshape(Ex, [3, 3])
    print('Ex: ', Ex)
    H_inv = np.matmul(K, Ex)
    print('H_inv: ', H_inv)
    H = np.mat(H_inv).I
    print('H: ', H)

    gt_pts, _pred, X, Y = hnet_transformation(gt_labels, pitch, 'inference')

    _loss = hnet_loss(gt_labels, pitch, 'loss')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        gt_pts_val, X_back, X_val, Y_val = sess.run([gt_pts, _pred, X, Y])
        print('gt_pts: ', gt_pts_val)
        print('X_back_proj: ', X_back)
        print('X_grd: ', X_val)
        print('Y_grd:', Y_val)
        loss_val = sess.run(_loss)
        print('loss:', loss_val)

