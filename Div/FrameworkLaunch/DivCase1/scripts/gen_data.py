#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import tensorflow as tf
import os
    
os.system("mkdir -p input")
os.system("mkdir -p output")

shape = [1, 1999]
minval = 1
maxval = 100

def gen_golden_data_simple():
    input_x = None
    input_y = None
    golden = None
    case_val = 4

    if case_val == 1:
        input_x = tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=tf.float16)
        input_y = tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=tf.float16)
    elif case_val == 2:
        input_x = tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=tf.float32)
        input_y = tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=tf.float32)
    elif case_val == 3:
        input_x = tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=tf.int8)
        input_y = tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=tf.int8)
    elif case_val == 4:
        input_x = tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=tf.int32)
        input_y = tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=tf.int32)

    golden = tf.raw_ops.Div(x=input_x, y=input_y)
    
    input_x.numpy().tofile("./input/input_x.bin")
    input_y.numpy().tofile("./input/input_y.bin")
    golden.numpy().tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

