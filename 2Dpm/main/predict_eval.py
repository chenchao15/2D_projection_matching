#!/usr/bin/env python

import startup

import os
import numpy as np
import tensorflow as tf

from util.app_config import config as app_config
from util.train import get_path

from main.predict import compute_predictions
from main.compute_chamfer_distance import run_eval

def compute_eval():
    cfg = app_config
    dataset = compute_predictions()
    result = run_eval(dataset)
    return result

def test_one_step(index):
    cfg = app_config
    train_dir = get_path(cfg)
    name = os.path.join(train_dir, 'chamfer_distance.txt')
    cfg.test_step = index
    result = compute_eval()
    with open(name, 'a+') as f:
        f.write(str(cfg.test_step) + ': ' + str(result) + '\n')

def main(_):
    cfg = app_config
    train_dir = get_path(cfg)
    res = []
    index = [200000]
    print('start testing ...')
    for i in index:    
        cfg.test_step = i
        result = compute_eval()
        res.append(result)
    with open(os.path.join(train_dir, 'chamfer_distance.txt'), 'w') as f:
        for i in res:
            f.write(str(i) + '\n')
    


if __name__ == '__main__':
    tf.app.run()
