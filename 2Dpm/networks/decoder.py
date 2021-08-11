import tensorflow as tf
import tensorflow.contrib.slim as slim


def model(inputs, outputs_all, cfg, is_training):
    num_points = cfg.pc_num_points

    init_stddev = cfg.pc_decoder_init_stddev
    w_init = tf.truncated_normal_initializer(stddev=init_stddev, seed=1)
    pts_raw = slim.fully_connected(inputs, num_points * 3,
                                   activation_fn=None,
                                   weights_initializer=w_init)

    pred_pts = tf.reshape(pts_raw, [pts_raw.shape[0], num_points, 3])
    pred_pts = tf.tanh(pred_pts)
    if cfg.pc_unit_cube:
        pred_pts = pred_pts / 2.0

    out = dict()
    out["xyz"] = pred_pts
    out["rgb"] = None

    return out
