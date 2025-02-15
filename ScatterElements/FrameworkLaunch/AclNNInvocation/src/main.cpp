/**
* @file main.cpp
*
* Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <cstdint>
#include <iostream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "acl/acl.h"
#include "op_runner.h"

#include "common.h"

bool g_isDevice = false;
int deviceId = 0;

OperatorDesc CreateOpDesc()
{
    // define operator
    OperatorDesc opDesc;

    // test-0
    // opDesc.axis = 0;
    // opDesc.reduce = "add";
    // std::vector<int64_t> shape_var {16};
    // std::vector<int64_t> shape_indices {8};
    // std::vector<int64_t> shape_updates {8};
    // aclDataType dataType = ACL_FLOAT16;
    // aclDataType indicesType = ACL_INT32;
    // aclFormat format = ACL_FORMAT_ND;

    // // test-1
    // opDesc.axis = 0;
    // opDesc.reduce = "add";
    // std::vector<int64_t> shape_var {64};
    // std::vector<int64_t> shape_indices {32};
    // std::vector<int64_t> shape_updates {32};
    // aclDataType dataType = ACL_FLOAT16;
    // aclDataType indicesType = ACL_INT32;
    // aclFormat format = ACL_FORMAT_ND;

    // test-2
    // opDesc.axis = 0;
    // opDesc.reduce = "add";
    // std::vector<int64_t> shape_var {3,5};
    // std::vector<int64_t> shape_indices {1,4};
    // std::vector<int64_t> shape_updates {1,4};
    // aclDataType dataType = ACL_FLOAT16;
    // aclDataType indicesType = ACL_INT32;
    // aclFormat format = ACL_FORMAT_ND;

    // test-3
    // opDesc.axis = 0;
    // opDesc.reduce = "multiply";
    // std::vector<int64_t> shape_var {3,5};
    // std::vector<int64_t> shape_indices {1,4};
    // std::vector<int64_t> shape_updates {1,4};
    // aclDataType dataType = ACL_INT32;
    // aclDataType indicesType = ACL_INT32;
    // aclFormat format = ACL_FORMAT_ND;

    // test-4
    // opDesc.axis = 0;
    // // opDesc.reduce = "add";
    // std::vector<int64_t> shape_var {3,5};
    // std::vector<int64_t> shape_indices {1,4};
    // std::vector<int64_t> shape_updates {1,4};
    // aclDataType dataType = ACL_INT32;
    // aclDataType indicesType = ACL_INT32;
    // aclFormat format = ACL_FORMAT_ND; 

    // test-5
    // opDesc.axis = 0;
    // opDesc.reduce = "multiply";
    // std::vector<int64_t> shape_var {3,5};
    // std::vector<int64_t> shape_indices {100,4};
    // std::vector<int64_t> shape_updates {100,4};
    // aclDataType dataType = ACL_UINT8;
    // aclDataType indicesType = ACL_INT32;
    // aclFormat format = ACL_FORMAT_ND; 

    // test-6 - 高维数据
    // opDesc.axis = 0;
    // opDesc.reduce = "add";
    // std::vector<int64_t> shape_var {2,3,4,5};
    // std::vector<int64_t> shape_indices {1,2,3,4};
    // std::vector<int64_t> shape_updates {1,2,3,4};
    // aclDataType dataType = ACL_FLOAT;
    // aclDataType indicesType = ACL_INT32;
    // aclFormat format = ACL_FORMAT_ND;

    // // test-7 - 大矩阵
    opDesc.axis = 1;
    opDesc.reduce = "multiply";
    std::vector<int64_t> shape_var {1,10};
    std::vector<int64_t> shape_indices {1,10};
    std::vector<int64_t> shape_updates {1,10};
    aclDataType dataType = ACL_INT32;
    aclDataType indicesType = ACL_INT32;
    aclFormat format = ACL_FORMAT_ND;

    // // test-8 - 一维数据
    // opDesc.axis = 0;
    // opDesc.reduce = "";  // None
    // std::vector<int64_t> shape_var {4};
    // std::vector<int64_t> shape_indices {2};
    // std::vector<int64_t> shape_updates {2};
    // aclDataType dataType = ACL_INT32;
    // aclDataType indicesType = ACL_INT32;
    // aclFormat format = ACL_FORMAT_ND;

    // // test-9 - 极限值范围fp16
    // opDesc.axis = 0;
    // opDesc.reduce = "add";
    // std::vector<int64_t> shape_var {5,5};
    // std::vector<int64_t> shape_indices {3,3};
    // std::vector<int64_t> shape_updates {3,3};
    // aclDataType dataType = ACL_FLOAT16;
    // aclDataType indicesType = ACL_INT32;
    // aclFormat format = ACL_FORMAT_ND;

    // // test-10 - int8最大值范围 - 溢出方式 与 pytorch 是相同的。当不把 gen_data.py 文件中的 tensor 转成 fp16 ，直接当作 int 处理时通过。
    // opDesc.axis = 0;
    // opDesc.reduce = "multiply";
    // std::vector<int64_t> shape_var {4,4};
    // std::vector<int64_t> shape_indices {2,2};
    // std::vector<int64_t> shape_updates {2,2};
    // aclDataType dataType = ACL_INT8;
    // aclDataType indicesType = ACL_INT32;
    // aclFormat format = ACL_FORMAT_ND;

    // // test-11 - 超高维
    // opDesc.axis = 0;
    // opDesc.reduce = "add";
    // std::vector<int64_t> shape_var {2,3,4,5,6,7};
    // std::vector<int64_t> shape_indices {1,2,3,4,5,6};
    // std::vector<int64_t> shape_updates {1,2,3,4,5,6};
    // aclDataType dataType = ACL_FLOAT;
    // aclDataType indicesType = ACL_INT32;
    // aclFormat format = ACL_FORMAT_ND;
    
    // test12
    // opDesc.axis = 1;
    // opDesc.reduce = "add";
    // std::vector<int64_t> shape_var {4,6};
    // std::vector<int64_t> shape_indices {4,2};
    // std::vector<int64_t> shape_updates {4,2};
    // aclDataType dataType = ACL_FLOAT;
    // aclDataType indicesType = ACL_INT32;
    // aclFormat format = ACL_FORMAT_ND;

    // test13
    // opDesc.axis = 2;
    // opDesc.reduce = "multiply";
    // std::vector<int64_t> shape_var {3,4,5};
    // std::vector<int64_t> shape_indices {3,4,2};
    // std::vector<int64_t> shape_updates {3,4,2};
    // aclDataType dataType = ACL_FLOAT16;
    // aclDataType indicesType = ACL_INT32;
    // aclFormat format = ACL_FORMAT_ND;

    // test14
    // opDesc.axis = 3;
    // opDesc.reduce = "";  // None
    // std::vector<int64_t> shape_var {2,3,4,8};
    // std::vector<int64_t> shape_indices {2,3,4,3};
    // std::vector<int64_t> shape_updates {2,3,4,3};
    // aclDataType dataType = ACL_FLOAT;
    // aclDataType indicesType = ACL_INT32;
    // aclFormat format = ACL_FORMAT_ND;

    // test15
    // opDesc.axis = 3;
    // opDesc.reduce = "";  // None
    // std::vector<int64_t> shape_var {2,3,4,8};
    // std::vector<int64_t> shape_indices {2,3,4,3};
    // std::vector<int64_t> shape_updates {4,5,6,7};
    // aclDataType dataType = ACL_FLOAT;
    // aclDataType indicesType = ACL_INT32;
    // aclFormat format = ACL_FORMAT_ND;

    // test16
    // opDesc.axis = 3;
    // opDesc.reduce = "";  // None
    // std::vector<int64_t> shape_var {7,8,9,10};
    // std::vector<int64_t> shape_indices {1,2,1,3};
    // std::vector<int64_t> shape_updates {4,5,6,7};
    
    // test17
    // opDesc.axis = 1;
    // opDesc.reduce = "";  // None
    // std::vector<int64_t> shape_var {7,8,9,10};
    // std::vector<int64_t> shape_indices {1,2,1,3};
    // std::vector<int64_t> shape_updates {1};

    // aclDataType dataType = ACL_FLOAT;
    // aclDataType indicesType = ACL_INT32;
    // aclFormat format = ACL_FORMAT_ND;
    

    opDesc.AddInputTensorDesc(dataType, shape_var.size(), shape_var.data(), format);
    opDesc.AddInputTensorDesc(indicesType, shape_indices.size(), shape_indices.data(), format);
    opDesc.AddInputTensorDesc(dataType, shape_updates.size(), shape_updates.data(), format);
    return opDesc;
}

bool SetInputData(OpRunner &runner)
{
    size_t fileSize = 0;
    ReadFile("../input/input_data.bin", fileSize, runner.GetInputBuffer<void>(0), runner.GetInputSize(0));
    ReadFile("../input/input_indices.bin", fileSize, runner.GetInputBuffer<void>(1), runner.GetInputSize(1));
    ReadFile("../input/input_updates.bin", fileSize, runner.GetInputBuffer<void>(2), runner.GetInputSize(2));
    INFO_LOG("Set input success");
    return true;
}

bool ProcessOutputData(OpRunner &runner)
{
    
    WriteFile("../output/output.bin", runner.GetInputBuffer<void>(0), runner.GetInputSize(0));

    INFO_LOG("Write output success");
    return true;
}

void DestoryResource()
{
    bool flag = false;
    if (aclrtResetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Reset device %d failed", deviceId);
        flag = true;
    }
    INFO_LOG("Reset Device success");
    if (aclFinalize() != ACL_SUCCESS) {
        ERROR_LOG("Finalize acl failed");
        flag = true;
    }
    if (flag) {
        ERROR_LOG("Destory resource failed");
    } else {
        INFO_LOG("Destory resource success");
    }
}

bool InitResource()
{
    std::string output = "../output";
    if (access(output.c_str(), 0) == -1) {
        int ret = mkdir(output.c_str(), 0700);
        if (ret == 0) {
            INFO_LOG("Make output directory successfully");
        }
        else {
            ERROR_LOG("Make output directory fail");
            return false;
        }
    }

    // acl.json is dump or profiling config file
    if (aclInit("../scripts/acl.json") != ACL_SUCCESS) {
        ERROR_LOG("acl init failed");
        return false;
    }

    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Set device failed. deviceId is %d", deviceId);
        (void)aclFinalize();
        return false;
    }
    INFO_LOG("Set device[%d] success", deviceId);

    // runMode is ACL_HOST which represents app is running in host
    // runMode is ACL_DEVICE which represents app is running in device
    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_SUCCESS) {
        ERROR_LOG("Get run mode failed");
        DestoryResource();
        return false;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    INFO_LOG("Get RunMode[%d] success", runMode);

    return true;
}

bool RunOp()
{
    // create op desc
    OperatorDesc opDesc = CreateOpDesc();

    // create Runner
    OpRunner opRunner(&opDesc);
    if (!opRunner.Init()) {
        ERROR_LOG("Init OpRunner failed");
        return false;
    }

    // Load inputs
    if (!SetInputData(opRunner)) {
        ERROR_LOG("Set input data failed");
        return false;
    }

    // Run op
    if (!opRunner.RunOp()) {
        ERROR_LOG("Run op failed");
        return false;
    }

    // process output data
    if (!ProcessOutputData(opRunner)) {
        ERROR_LOG("Process output data failed");
        return false;
    }

    INFO_LOG("Run op success");
    return true;
}

int main(int argc, char **argv)
{
    if (!InitResource()) {
        ERROR_LOG("Init resource failed");
        return FAILED;
    }
    INFO_LOG("Init resource success");

    if (!RunOp()) {
        DestoryResource();
        return FAILED;
    }

    DestoryResource();

    return SUCCESS;
}
