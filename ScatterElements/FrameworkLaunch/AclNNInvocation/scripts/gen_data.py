import numpy as np
import os
import torch


def get_golden(data, indices, updates, axis=0,reduction='none'):
    # data = data.astype(np.float32)
    indices = indices.astype(np.int64)
    # updates = updates.astype(np.float32)
    if reduction == "add":
        res = torch.scatter_add(torch.from_numpy(data), axis, torch.from_numpy(indices), torch.from_numpy(updates))
        # res = torch.scatter(torch.from_numpy(data), axis, torch.from_numpy(indices), torch.from_numpy(updates), reduce='add')
    elif reduction == "multiply":
        res = torch.scatter(torch.from_numpy(data), axis, torch.from_numpy(indices), torch.from_numpy(updates), reduce='multiply')
    else:
        if torch.from_numpy(updates).shape == torch.Size([1]):
            res = torch.scatter(torch.from_numpy(data), axis, torch.from_numpy(indices), torch.from_numpy(updates).item())
        else:
            res = torch.scatter(torch.from_numpy(data), axis, torch.from_numpy(indices), torch.from_numpy(updates))
    # res = res.numpy().astype(np.float16)
    res = res.numpy()
    return res

def gen_golden_data_simple():
    os.system("mkdir -p input")
    os.system("mkdir -p output")

    # test0
    # input_data = np.random.uniform(-1, 1, [16]).astype(np.float16)
    # input_indices = np.random.uniform(0, 16, [8]).astype(np.int32)
    # input_updates = np.random.uniform(-10, 10, [8]).astype(np.float16)
    # axis = 0
    # reduction = 'add'

    # # test1
    # input_data = np.random.uniform(-1, 1, [64]).astype(np.float16)
    # input_indices = np.random.uniform(0, 60, [32]).astype(np.int32)
    # input_updates = np.random.uniform(-10, 10, [32]).astype(np.float16)
    # axis = 0
    # reduction = 'add'

    # test2 
    # input_data = np.random.uniform(-1, 1, [3, 5]).astype(np.float16)
    # input_indices = np.random.uniform(0, 3, [1,4]).astype(np.int32)
    # input_updates = np.random.uniform(-10, 10, [1,4]).astype(np.float16)
    # axis = 0
    # reduction = 'add'

    # test3
    # input_data = np.random.uniform(-10,10 , [3, 5]).astype(np.int32)
    # input_indices = np.random.uniform(0, 3, [1,4]).astype(np.int32)
    # input_updates = np.random.uniform(-10, 10, [1,4]).astype(np.int32)
    # axis = 0
    # reduction = 'multiply'

    # test4
    # input_data = np.random.uniform(-10,10 , [3, 5]).astype(np.int32)
    # input_indices = np.random.uniform(0, 3, [1,4]).astype(np.int32)
    # input_updates = np.random.uniform(-10, 10, [1,4]).astype(np.int32)
    # axis = 0
    # reduction = None

    # test5
    # input_data = np.ones([3, 5]).astype(np.uint8)
    # input_indices = np.random.uniform(0, 3, [100,4]).astype(np.int32)
    # input_updates = np.random.uniform(0, 1000, [100,4]).astype(np.uint8)
    # axis = 0
    # reduction = 'multiply'

    # test6 - 测试高维数据
    # input_data = np.random.uniform(-1, 1, [2,3,4,5]).astype(np.float32)
    # input_indices = np.random.uniform(0, 2, [1,2,3,4]).astype(np.int32)  
    # input_updates = np.random.uniform(-1, 1, [1,2,3,4]).astype(np.float32)
    # axis = 0
    # reduction = 'add'

    # # test7 - 测试大矩阵
    input_data = np.random.uniform(10000, 100000000, [1, 10]).astype(np.int32)
    input_indices = np.random.uniform(0, 10, [1, 10]).astype(np.int32)
    input_updates = np.random.uniform(10000, 100000000, [1, 10]).astype(np.int32) 
    axis = 1
    reduction = 'multiply'

    # # test8 - 测试一维数据
    # input_data = np.random.uniform(1, 10, [4]).astype(np.int32)
    # input_indices = np.random.uniform(0, 4, [2]).astype(np.int32)
    # input_updates = np.random.uniform(1, 10, [2]).astype(np.int32)
    # axis = 0
    # reduction = None

    # # test9 - 测试极限值范围fp16
    # input_data = np.random.uniform(-1000.0, 1000.0, [5,5]).astype(np.float16)
    # input_indices = np.random.uniform(0, 5, [3,3]).astype(np.int32)
    # input_updates = np.random.uniform(-1000.0, 1000.0, [3,3]).astype(np.float16)
    # axis = 0
    # reduction = 'add'

    # # test10 - 测试int8最大值范围  溢出方式 与 pytorch 是相同的。当不把 gen_data.py 文件中的 tensor 转成 fp16 ，直接当作 int 处理时通过。
    # input_data = np.random.uniform(-128, 127, [4,4]).astype(np.int8) 
    # input_indices = np.random.uniform(0, 4, [2,2]).astype(np.int32)
    # input_updates = np.random.uniform(-128, 127, [2,2]).astype(np.int8)
    # axis = 0
    # reduction = 'multiply'

    # # test11 - 测试超高维
    # input_data = np.random.uniform(-1, 1, [2,3,4,5,6,7]).astype(np.float32)
    # input_indices = np.random.uniform(0, 2, [1,2,3,4,5,6]).astype(np.int32)
    # input_updates = np.random.uniform(-1, 1, [1,2,3,4,5,6]).astype(np.float32)
    # axis = 0 
    # reduction = 'add'

    # test12 - axis=1, 2D数据
    # input_data = np.random.uniform(-1, 1, [4,6]).astype(np.float32)
    # input_indices = np.random.uniform(0, 6, [4,2]).astype(np.int32)  # indices范围要对应axis=1的维度大小
    # input_updates = np.random.uniform(-1, 1, [4,2]).astype(np.float32)
    # axis = 1
    # reduction = 'add'

    # # test13 - axis=2, 3D数据
    # input_data = np.random.uniform(-1, 1, [3,4,5]).astype(np.float16)
    # input_indices = np.random.uniform(0, 5, [3,4,2]).astype(np.int32)  # indices范围要对应axis=2的维度大小
    # input_updates = np.random.uniform(-1, 1, [3,4,2]).astype(np.float16)
    # axis = 2
    # reduction = 'multiply'

    # # test14 - axis=3, 4D数据
    # input_data = np.random.uniform(-100, 100, [2,3,4,8]).astype(np.float32)
    # input_indices = np.random.uniform(0, 8, [2,3,4,3]).astype(np.int32)  # indices范围要对应axis=3的维度大小
    # input_updates = np.random.uniform(-100, 100, [2,3,4,3]).astype(np.float32)
    # axis = 3
    # reduction = None

    # # test15 - axis=3, 4D数据,  indices.shape != updates.shape
    # input_data = np.random.uniform(-100, 100, [2,3,4,8]).astype(np.float32)
    # input_indices = np.random.uniform(0, 8, [2,3,4,3]).astype(np.int32)  # indices范围要对应axis=3的维度大小
    # input_updates = np.random.uniform(-100, 100, [4,5,6,7]).astype(np.float32)
    # axis = 3
    # reduction = None


    # # test16 - axis=3, 4D数据,  indices.shape != updates.shape
    # input_data = np.random.uniform(-100, 100, [7,8,9,10]).astype(np.float32)
    # input_indices = np.random.uniform(0, 8, [1,2,1,3]).astype(np.int32)  # indices范围要对应axis=3的维度大小
    # input_updates = np.random.uniform(-100, 100, [4,5,6,7]).astype(np.float32)
    # axis = 3
    # reduction = None

    # # test17 - axis=1, 4D数据,  indices.shape != updates.shape
    # input_data = np.random.uniform(-100, 100, [7,8,9,10]).astype(np.float32)
    # input_indices = np.random.uniform(0, 8, [1,2,1,3]).astype(np.int32)  # indices范围要对应axis=3的维度大小
    # input_updates = np.random.uniform(-100, 100, [1]).astype(np.float32)
    # axis = 1
    # reduction = None

    input_data.tofile("./input/input_data.bin")
    input_indices.tofile("./input/input_indices.bin")
    input_updates.tofile("./input/input_updates.bin")
    print(input_data)
    print(input_indices)
    print(input_updates)
    golden = get_golden(input_data, input_indices, input_updates,axis=axis,reduction=reduction)
    print(golden)
    golden.tofile("./output/golden.bin")


def calc_expect_func(data, indices, updates, y, axis,reduction):
    input_data = data["value"]
    input_indices = indices["value"]
    input_updates = updates["value"] 
    res = get_golden(input_data, input_indices, input_updates,axis,reduction)

    return [res,]


if __name__ == "__main__":
    gen_golden_data_simple()
