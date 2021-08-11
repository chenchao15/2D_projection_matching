#!/usr/bin/env python

import startup

import tensorflow as tf

from main import train
from main.predict_eval import compute_eval

def main(_):
    train.train()
    compute_eval()

if __name__ == '__main__':
    tf.app.run()
