#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import tensorflow as tf
import os
import argparse

tf.random.set_seed(29)
os.system("mkdir -p input")
os.system("mkdir -p output")

test_cases = [
    {"shape": [64], "minval": -10, "maxval": 10, "dtype": tf.float16},
    {"shape": [1024, 1024], "minval": -1000, "maxval": 1000, "dtype": tf.float16},
    {"shape": [64], "minval": -10, "maxval": 10, "dtype": tf.float32},
    {"shape": [1024, 1024], "minval": -1000, "maxval": 1000, "dtype": tf.float32},
    {"shape": [64], "minval": -128, "maxval": 127, "dtype": tf.int8},
    {"shape": [1024, 1024], "minval": -128, "maxval": 127, "dtype": tf.int8},
    {"shape": [9,100,5,1024], "minval": -128, "maxval": 127, "dtype": tf.int8},
    {"shape": [1,1,55000], "minval": -128, "maxval": 127, "dtype": tf.int8},
]

def gen_golden_data_simple(shape, minval, maxval, dtype):

    input_x = tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=dtype)
    input_x.numpy().tofile("./input/input_x.bin")
    
    tmp = tf.random.uniform(shape, minval=0, maxval=0.01, dtype=dtype)
    input_y = input_x - tmp
    input_y.numpy().tofile("./input/input_y.bin")

    golden = tf.raw_ops.NotEqual(x=input_x, y=input_y)
    golden.numpy().tofile("./output/golden.bin")
    # print(input_x)
    # print(input_y)
    # print(golden)


def gen_golden_data_simple_int8(shape, minval, maxval, dtype):

    # 先用 int32 生成随机数
    input_x = tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=tf.int32)
    input_y = tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=tf.int32)
    
    # 确保 input_y <= input_x
    input_y = tf.minimum(input_x, input_y)
    
    # 将 int32 转换为 int8
    input_x = tf.cast(tf.clip_by_value(input_x, -128, 127), tf.int8)
    input_y = tf.cast(tf.clip_by_value(input_y, -128, 127), tf.int8)
    
    # 保存为二进制文件
    input_x.numpy().tofile("./input/input_x.bin")
    input_y.numpy().tofile("./input/input_y.bin")

    # 生成比较结果
    golden = tf.raw_ops.NotEqual(x=input_x, y=input_y)
    golden.numpy().tofile("./output/golden.bin")
    
    # print("Input X (int8):", input_x.numpy())
    # print("Input Y (int8):", input_y.numpy())
    # print("Golden (NotEqual):", golden.numpy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_id", type=int, required=True, help="test id")
    args = parser.parse_args()
    test_id = args.test_id
    test_case = test_cases[test_id]
    shape = test_case["shape"]
    minval = test_case["minval"]
    maxval = test_case["maxval"]
    dtype = test_case["dtype"]

    if dtype == tf.int8:
        gen_golden_data_simple_int8(shape, minval, maxval, dtype)  # Call the int8 specific function
    else:
        gen_golden_data_simple(shape, minval, maxval, dtype)  # Call the general function for other types