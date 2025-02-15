#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import tensorflow as tf
import os
    
tf.random.set_seed(29)
os.system("mkdir -p input")
os.system("mkdir -p output")
# test - 1 
# shape = [64]
# minval = 10
# maxval = -10
# dtype = tf.float16
# sucess

# test - 2
# shape = [1024,1024]
# minval = -1000
# maxval = 1000
# dtype = tf.float16
# sucess

# test - 2.1
# shape = [2,3,4,5,6,7]
# minval = -1
# maxval = 1
# dtype = tf.float16
# sucess

# test - 2.2
# shape = [1]
# minval = -1000
# maxval = 1000
# dtype = tf.float16
# sucess

# test - 3
# shape = [64]
# minval = -10
# maxval = 10
# dtype = tf.float32
# sucess


# test - 4
# shape = [1024,1024]
# minval = -1000
# maxval = 1000
# dtype = tf.float32
# sucess

# test - 5
# shape = [64]
# minval = -10
# maxval = 10
# dtype = tf.int32
# sucess

# test - 5
# shape = [1024,1024]
# minval = -1000
# maxval = 1000
# dtype = tf.int32
# sucess

# test - 5.1
# shape = [2,3,4,5,6,7]
# minval = -1000
# maxval = 1000
# dtype = tf.int32
# sucess


# test - 6
# shape = [64]
# minval = -128  # int8 的最小值
# maxval = 127   # int8 的最大值
# dtype = tf.int8

# test - 7
# shape = [1024,1024]
# minval = -128  # int8 的最小值
# maxval = 127   # int8 的最大值
# dtype = tf.int8

# test - 8
# shape = [9,100,5,1024]
# minval = -128  # int8 的最小值
# maxval = 127   # int8 的最大值
# dtype = tf.int8
# sucess

# test - 9
# shape = [1,1,55000]
# minval = -128  # int8 的最小值
# maxval = 127   # int8 的最大值
# dtype = tf.int8
# sucess

# test - 9.1
# shape = [2,1,55000]
# minval = -1000  
# maxval = 1000  
# dtype = tf.int8
# sucess

# test - 9.2
# shape = [13,17,19]
# minval = -1000  
# maxval = 1000  
# dtype = tf.int8
# sucess

# test - 10 
# shape = [64,1,32]
# shape1 = [64,32,32]
# minval = 10
# maxval = -10
# dtype = tf.float16
# sucess
# test - 11
# shape = [1, 64, 32]
# shape1 = [64, 1, 32]
# minval = -10
# maxval = 10
# dtype = tf.float16
# # sucess

# # test - 12
shape = [2, 1, 3]
shape1 = [2, 2, 3]
minval = -1
maxval = 1
dtype = tf.float16
# # sucess


def gen_golden_data_simple():

    input_x = tf.random.uniform(shape1, minval=minval, maxval=maxval, dtype=dtype)
    input_x.numpy().tofile("./input/input_x.bin")
    
    input_y = tf.random.uniform(shape1, minval=minval, maxval=maxval, dtype=dtype)
    input_y = tf.random.uniform(shape1, minval=minval, maxval=maxval, dtype=dtype)
    input_y = tf.minimum(input_x, input_y)  

    input_x_np = input_x.numpy()
    input_y_np = input_y.numpy()
    input_x_np[:,:,:] = 5
    input_y_np[:,:,:] = 5
    input_x = tf.convert_to_tensor(input_x_np)
    input_y = tf.convert_to_tensor(input_y_np)

    input_y.numpy().tofile("./input/input_y.bin")
    golden = tf.raw_ops.NotEqual(x=input_x, y=input_y)
    golden.numpy().tofile("./output/golden.bin")
    print(input_x)
    print(input_y)
    print(golden)

def gen_golden_data_simple_broadcast():

    input_x = tf.random.uniform(shape1, minval=minval, maxval=maxval, dtype=dtype)    
    input_y = tf.random.uniform(shape1, minval=minval, maxval=maxval, dtype=dtype)
    input_y = tf.maximum(input_x, input_y) 
    input_x = tf.slice(input_x, begin=[0, 0, 0], size=shape)

    input_x.numpy().tofile("./input/input_x.bin")
    input_y.numpy().tofile("./input/input_y.bin")
    golden = tf.raw_ops.NotEqual(x=input_x, y=input_y)
    golden.numpy().tofile("./output/golden.bin")
    print(input_x)
    print(input_y)
    print(golden)

def gen_golden_data_simple_int():
    # 生成第一个随机整数张量
    input_x = tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=dtype)
    input_x.numpy().tofile("./input/input_x.bin")
    
    # 生成第二个随机整数张量，确保值略小于 input_x
    input_y = tf.random.uniform(shape1, minval=minval, maxval=maxval, dtype=dtype)
    input_y = tf.minimum(input_x, input_y)  
    input_y.numpy().tofile("./input/input_y.bin")

    # 生成比较结果
    golden = tf.raw_ops.NotEqual(x=input_x, y=input_y)
    golden.numpy().tofile("./output/golden.bin")
    
    print("Input X (int32):", input_x.numpy())
    print("Input Y (int32):", input_y.numpy())
    print("Golden (NotEqual):", golden.numpy())


def gen_golden_data_simple_int8():
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
    
    print("Input X (int8):", input_x.numpy())
    print("Input Y (int8):", input_y.numpy())
    print("Golden (NotEqual):", golden.numpy())


if __name__ == "__main__":
    # if dtype == tf.int8:
    #     gen_golden_data_simple_int8()
    # elif dtype == tf.int32:
    #     gen_golden_data_simple_int()
    # else:
    #     gen_golden_data_simple()
    gen_golden_data_simple_broadcast()