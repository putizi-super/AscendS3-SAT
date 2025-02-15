#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import os

def gen_golden_data_simple():
    input_x = None
    input_y = None
    golden = None
    case_val = 2
    if case_val == 1:
        input_x = np.random.uniform(1, 100, [1, 1999]).astype(np.float16)
        input_y = np.random.uniform(1, 100, [1, 1999]).astype(np.float16)
        golden = (input_x / input_y).astype(np.float16)
    elif case_val == 2:
        input_x = np.random.uniform(1, 100, [1, 1999]).astype(np.float32)
        input_y = np.random.uniform(1, 100, [1, 1999]).astype(np.float32)
        golden = (input_x / input_y).astype(np.float32)
    elif case_val == 3:
        input_x = np.random.uniform(1, 100, [1, 1999]).astype(np.int8)
        input_y = np.random.uniform(1, 100, [1, 1999]).astype(np.int8)
        golden = (input_x / input_y).astype(np.int8)
    elif case_val == 4:
        input_x = np.random.uniform(1, 100, [1, 1999]).astype(np.int32)
        input_y = np.random.uniform(1, 100, [1, 1999]).astype(np.int32)
        golden = (input_x / input_y).astype(np.int32)
    
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
