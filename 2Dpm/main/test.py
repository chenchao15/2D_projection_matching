#!/usr/bin/env python
import startup
import tensorflow as tf
from run.predict_eval import test_one_step
from util.app_config import config as app_config

def test():
    cfg = app_config
    global_step_val = cfg.test_step
    test_one_step(global_step_val)


def main(_):
    test()


if __name__ == '__main__':
    tf.app.run()
