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
from config import global_config

CFG = global_config.cfg
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
max_dist = CFG.TRAIN.MAX_DIST


def hnet_loss(gt_pts, pitch, name):
    """
    :param gt_pts: image points [x, y, 1]
    :param H: homograph between image and ground plane [[a, b, c], [0, d, e], [0, f, 1]]
    :param name:
    :return:
    """
    with tf.variable_scope(name):
        # modified network is supposed to output pitch angle only
        gt_pts = tf.cast(gt_pts, dtype=tf.float64)
        pitch = tf.cast(pitch, dtype=tf.float64)
        # compute homograph matrix
        c1 = tf.cos(pitch)
        s1 = tf.sin(pitch)

        K = tf.constant([fx, 0.0, cx,
                         0.0, fy, cy,
                         0.0, 0.0, 1.0], dtype=tf.float64)
        K = tf.reshape(K, shape=[3, 3])
        Ex = tf.stack(
            [tf.constant(1.0, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64),
             tf.constant(0.0, dtype=tf.float64), s1, Tz * c1,
             tf.constant(0.0, dtype=tf.float64), c1, Tz * s1])
        Ex = tf.reshape(Ex, shape=[3, 3])
        H_inv = tf.matmul(K, Ex)
        H = tf.matrix_inverse(H_inv)

        gt_pts = tf.transpose(gt_pts)
        pts_projects = tf.matmul(H, gt_pts)

        # second-order polynomial fitting
        X = tf.multiply(pts_projects[0, :], tf.reciprocal(pts_projects[2, :]))
        Y = tf.multiply(pts_projects[1, :], tf.reciprocal(pts_projects[2, :]))
        X = tf.transpose(X)
        Y = tf.transpose(Y)

        Y_valid = tf.clip_by_value(Y, -max_dist, max_dist)
        Y_stack_valid = tf.stack([tf.pow(Y_valid, 3), tf.pow(Y_valid, 2), Y_valid, tf.ones_like(Y_valid)], axis=1)
        # X_valid = tf.gather(X, valid_indices)
        X_valid = tf.clip_by_value(X, -max_dist, max_dist)
        tmp_mat = tf.matmul(tf.transpose(Y_stack_valid), Y_stack_valid)

        w = tf.matmul(tf.matmul(tf.matrix_inverse(tmp_mat),
                                tf.transpose(Y_stack_valid)), tf.expand_dims(X_valid, -1))

        # compute re-projection error in image coordinates
        Y_One = tf.ones_like(Y_valid)
        x_preds = tf.matmul(Y_stack_valid, w)
        preds = tf.transpose(tf.stack([tf.squeeze(x_preds, -1), Y_valid, Y_One], axis=1))
        pts_trans_back = tf.matmul(H_inv, preds)
        X_trans_back = tf.multiply(pts_trans_back[0, :], tf.reciprocal(pts_trans_back[2, :]))
        losses = tf.pow(gt_pts[0, :] - X_trans_back, 2)
        # loss = tf.reduce_mean(tf.log(tf.add(losses, 1.00001)))
        loss = tf.reduce_mean(losses)
    return tf.cast(loss, tf.float32)


def hnet_transformation(gt_pts, pitch, name):
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

        K = tf.constant([fx, 0.0, cx,
                         0.0, fy, cy,
                         0.0, 0.0, 1.0], dtype=tf.float64)
        K = tf.reshape(K, shape=[3, 3])
        Ex = tf.stack(
            [tf.constant(1.0, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64),
             tf.constant(0.0, dtype=tf.float64), s1, Tz * c1,
             tf.constant(0.0, dtype=tf.float64), c1, Tz * s1])
        Ex = tf.reshape(Ex, shape=[3, 3])
        H_inv = tf.matmul(K, Ex)
        H = tf.matrix_inverse(H_inv)

        gt_pts = tf.transpose(gt_pts)
        pts_projects = tf.matmul(H, gt_pts)

        # second-order polynomial fitting
        X = tf.multiply(pts_projects[0, :], tf.reciprocal(pts_projects[2, :]))
        Y = tf.multiply(pts_projects[1, :], tf.reciprocal(pts_projects[2, :]))
        X = tf.transpose(X)
        Y = tf.transpose(Y)

        # apply fitting to Y samples within valid range
        # mask = tf.logical_and(Y > -max_dist, Y < max_dist)
        # valid_indices = tf.where(mask)[:, 0]
        # Y_valid = tf.gather(Y, valid_indices)
        Y_valid = tf.clip_by_value(Y, -max_dist, max_dist)
        Y_stack_valid = tf.stack([tf.pow(Y_valid, 3), tf.pow(Y_valid, 2), Y_valid, tf.ones_like(Y_valid)], axis=1)
        # X_valid = tf.gather(X, valid_indices)
        X_valid = tf.clip_by_value(X, -max_dist, max_dist)
        tmp_mat = tf.matmul(tf.transpose(Y_stack_valid), Y_stack_valid)

        w = tf.matmul(tf.matmul(tf.matrix_inverse(tmp_mat),
                                tf.transpose(Y_stack_valid)), tf.expand_dims(X_valid, -1))

        # compute re-projection error in image coordinates
        Y_One = tf.ones_like(Y_valid)
        x_preds = tf.matmul(Y_stack_valid, w)
        preds = tf.transpose(tf.stack([tf.squeeze(x_preds, -1), Y_valid, Y_One], axis=1))
        pts_trans_back = tf.matmul(H_inv, preds)
        X_trans_back = tf.multiply(pts_trans_back[0, :], tf.reciprocal(pts_trans_back[2, :]))
        X_trans_back_valid = X_trans_back
    return gt_pts, X_trans_back, X, Y, X_valid, Y_valid, tmp_mat, w, X_trans_back_valid, H_inv, H


def hnet_loss_np(gt_pts, pitch):
    """
    :param gt_pts: image points [x, y, 1]
    :param H: homograph between image and ground plane [[a, b, c], [0, d, e], [0, f, 1]]
    :param name:
    :return:
    """

    # modified network is supposed to output pitch angle only
    # compute homograph matrix
    c1 = np.cos(pitch)
    s1 = np.sin(pitch)

    K = np.array([fx, 0.0, cx,
                  0.0, fy, cy,
                  0.0, 0.0, 1.0])
    K = np.reshape(K, [3, 3])
    Ex = np.array([1.0, 0.0, 0.0,
                   0.0, s1, Tz * c1,
                   0.0, c1, Tz * s1])
    Ex = np.reshape(Ex, [3, 3])
    H_inv = np.matmul(K, Ex)
    H = np.mat(H_inv).I

    gt_pts = np.transpose(gt_pts)
    pts_projects = np.matmul(H, gt_pts)

    # second-order polynomial fitting
    X = np.multiply(pts_projects[0, :], np.reciprocal(pts_projects[2, :]))
    Y = np.multiply(pts_projects[1, :], np.reciprocal(pts_projects[2, :]))
    X = np.transpose(X)
    Y = np.transpose(Y)

    # apply fitting to Y samples within valid range
    # mask = np.logical_and(Y > -max_dist, Y < max_dist)
    # [valid_indices, _] = np.where(mask)
    # Y_valid = Y[valid_indices, :]
    Y_valid = np.clip(Y, -max_dist, max_dist)
    Y_stack_valid = np.column_stack([np.power(Y_valid, 3), np.power(Y_valid, 2), Y_valid, np.ones_like(Y_valid)])
    # X_valid = X[valid_indices, :]
    X_valid = np.clip(X, -max_dist, max_dist)
    tmp_mat = np.mat(np.matmul(np.transpose(Y_stack_valid), Y_stack_valid)).I
    w = np.matmul(np.matmul(tmp_mat, np.transpose(Y_stack_valid)), X_valid)

    # compute re-projection error in image coordinates
    Y_One = np.ones_like(Y, np.float32)
    # Y_stack = np.column_stack([np.power(Y, 3), np.power(Y, 2), Y, Y_One])
    x_preds = np.matmul(Y_stack_valid, w)
    preds = np.transpose(np.column_stack([x_preds, Y_valid, Y_One]))
    pts_trans_back = np.matmul(H_inv, preds)
    X_trans_back = np.multiply(pts_trans_back[0, :], np.reciprocal(pts_trans_back[2, :]))
    # X_trans_back_valid = X_trans_back[0, valid_indices]
    X_trans_back_valid = X_trans_back
    losses = np.power(gt_pts[0, :] - X_trans_back_valid, 2)
    # loss = np.mean(np.log(losses + 1.00001))
    loss = np.mean(losses)
    return loss, X_trans_back, X, Y, X_valid, Y_valid, tmp_mat, w, X_trans_back_valid


if __name__ == '__main__':
    pitch = -22.77807426
    # org_input = [[0.0, 63.0, 1.0], [8.0, 59.0, 1.0], [16.0, 55.0, 1.0],
    #              [24.0, 51.0, 1.0], [32.0, 47.0, 1.0], [40.0, 43.0, 1.0],
    #              [63.0, 32.0, 1.0]]
    org_input = [[52.7, 26.66666667, 1.],
                 [49.3, 27.55555556, 1.],
                 [45.9, 28.44444444, 1.],
                 [42.5, 29.33333333, 1.],
                 [39.1, 30.22222222, 1.],
                 [35.7, 31.11111111, 1.],
                 [32.3, 32., 1.],
                 [28.9, 32.88888889, 1.],
                 [25.5, 33.77777778, 1.],
                 [22.1, 34.66666667, 1.],
                 [18.7, 35.55555556, 1.],
                 [15.3, 36.44444444, 1.],
                 [11.9, 37.33333333, 1.],
                 [8.5, 38.22222222, 1.],
                 [5.1, 39.11111111, 1.],
                 [1.7, 40., 1.]]
    gt_labels_np = np.array(org_input)
    gt_labels_np = np.reshape(gt_labels_np, [-1, 3])
    gt_labels = tf.constant(org_input, shape=gt_labels_np.shape)
    gt_labels64 = tf.constant(org_input, dtype=tf.float64, shape=gt_labels_np.shape)
    # gt_labels_np = gt_labels_np[:2, :]
    # gt_labels = gt_labels[:2, :]
    # gt_labels64 = gt_labels64[:2, :]

    c1 = np.cos(pitch)
    s1 = np.sin(pitch)
    print('fx: ', fx)
    print('fy: ', fy)

    K = np.array([fx, 0.0, cx,
                  0.0, fy, cy,
                  0.0, 0.0, 1.0])
    K = np.reshape(K, [3, 3])
    print('K: ', K)
    Ex = np.array([1.0, 0.0, 0.0,
                   0.0, s1, c1 * Tz,
                   0.0, c1, s1 * Tz])
    Ex = np.reshape(Ex, [3, 3])
    print('Ex: ', Ex)
    H_inv = np.matmul(K, Ex)
    print('H_inv: ', H_inv)
    H = np.mat(H_inv).I
    print('H: ', H)

    # compute transformation using float64 to compare with numpy implementation
    gt_pts, X_trans_back, X, Y, X_valid, Y_valid, tmp_mat, w, X_trans_back_valid, H_inv, H = \
        hnet_transformation(gt_labels64, tf.constant(pitch, dtype=tf.float64), 'inference')

    # actual training will use float32 for efficiency
    _loss = hnet_loss(gt_labels, tf.constant(pitch), 'loss')

    print('\n results of numpy implementation')
    loss_np, X_back_np, X_val_np, Y_val_np, X_valid_val_np, Y_valid_val_np, tmp_mat_val_np, w_val_np, X_back_valid_val_np = \
        hnet_loss_np(gt_labels_np, pitch)
    print('X_grd_np: ', X_val_np)
    print('Y_grd_np: ', Y_val_np)
    print('X_grd_valid_np: ', X_valid_val_np)
    print('Y_grd_valid_np: ', Y_valid_val_np)
    print('tmp_mat_np: ', tmp_mat_val_np)
    print('w_np: ', w_val_np)
    print('X_back_np: ', X_back_np)
    print('X_back_valid_np: ', X_back_valid_val_np)
    print('loss_np:', loss_np)

    print('\n results of tf float64 implementation')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        gt_pts_val, X_back, X_val, Y_val, X_valid_val, Y_valid_val, tmp_mat_val, w_val, X_back_valid_val, H_inv_val, H_val = \
            sess.run([gt_pts, X_trans_back, X, Y, X_valid, Y_valid, tmp_mat, w, X_trans_back_valid, H_inv, H])
        print('H_inv: ', H_inv_val)
        print('H: ', H_val)
        print('gt_pts: ', gt_pts_val)
        print('X_grd: ', X_val)
        print('Y_grd: ', Y_val)
        print('X_grd_valid: ', X_valid_val)
        print('Y_grd_valid: ', Y_valid_val)
        print('tmp_mat: ', tmp_mat_val)
        print('w: ', w_val)
        print('X_back: ', X_back)
        print('X_back_valid: ', X_back_valid_val)

        loss_val = sess.run(_loss)
        print('loss float32:', loss_val)

    print('\n compare the error caused by TF by using float64 compared to numpy float64')
    print('diff in inverse mat: ', np.subtract(tmp_mat_val, tmp_mat_val_np))
    print('diff in w: ', np.subtract(w_val, w_val_np))
    print('diff in x_back: ', np.subtract(X_back_valid_val, X_back_valid_val_np))
