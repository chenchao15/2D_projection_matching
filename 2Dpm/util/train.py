import tensorflow as tf
import os

def get_trainable_variables(scopes):
    is_trainable = lambda x: x in tf.trainable_variables()

    var_list = []

    for scope in scopes:
        var_list_raw = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        var_list_scope = list(filter(is_trainable, var_list_raw))
        var_list.extend(var_list_scope)

    return var_list


def get_learning_rate_origin(cfg, global_step, add_summary=True):
    if not cfg.decay:
        step_val = cfg.learning_rate_step * cfg.max_number_of_steps
        global_step = tf.cast(global_step, tf.float32)
        lr = tf.where(tf.less(global_step, step_val), cfg.learning_rate, cfg.learning_rate_2)
        if add_summary:
            tf.contrib.summary.scalar("learning_rate", lr)
    else:
        step = cfg.each_steps
        global_step = tf.cast(global_step, tf.float32)
        n = tf.floor(tf.divide(global_step, step))
        bilv = tf.pow(cfg.decay_rate, n)
        lr = cfg.learning_rate * bilv
    return lr

def get_learning_rate(cfg, global_step, add_summary=True):
    def ccc(global_step):
        step = cfg.each_steps
        global_steps = tf.cast(global_step, tf.float32)
        n = tf.floor(tf.divide(global_steps, step))
        bilv = tf.pow(0.81, n)
        lr = cfg.learning_rate * bilv
        return lr
    lr = ccc(global_step)
    # lr = tf.where(tf.less(global_step, 100000), cfg.learning_rate, ccc(global_step))
    return lr

def get_path(cfg):
    base_dir = cfg.checkpoint_dir
    name_dir = f"lr-{cfg.learning_rate}_dataset-{cfg.synth_set}_pointn-{cfg.pc_num_points}_gtp-{cfg.gt_point_n}"
    if cfg.decay:
        name_dir += f"_lrdecay-{cfg.decay_rate}"
    return os.path.join(base_dir, name_dir)
