
#include "scatter_elements_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h" // 引入平台相关头文件
#include <cstddef> // 引入cstddef头文件，定义了size_t等类型
#include <cstdint> // 引入cstdint头文件，定义了固定宽度整数类型
#include <cstring> // 引入cstring头文件，提供C风格字符串处理功能


constexpr uint32_t BLOCK_SIZE = 32; 

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ScatterElementsTilingData tiling;
    uint64_t ubSize; // 定义ubSize变量
    uint32_t bufferNum = 16; // 定义16个buffer
    auto ascendcPlatform =platform_ascendc::PlatformAscendC(context->GetPlatformInfo()); // 获取AscendC平台信息
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize); // 获取核心内存大小
    uint32_t dataType = context->GetInputDesc(0)->GetDataType(); // 获取输入数据类型 
    // 获得 var 的维度信息
    int32_t numdims = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    // 获得 三个输入 tensor 的shape信息 以及元素个数
    // // 如需要使用系统workspace需要调用GetLibApiWorkSpaceSize获取系统workspace的大小。
    // auto ascendcPlatform = platform_ascendc:: PlatformAscendC(context->GetPlatformInfo());
    // uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    int32_t shape[192];
    int32_t size[3] = {1 ,1 ,1 };
    int32_t ndims[3] = {0 ,0, 0};
    for(int32_t k = 0; k < 3; ++k){
        int32_t *ss = &shape[k * 64];
        const gert::StorageShape* shape = context->GetInputShape(k); // 分别获得三个输入的shape 信息
        for (int32_t i = 0; i < shape->GetStorageShape().GetDimNum(); i++) { // 获得每个输入的shape信息比如 [1,2,3,4] 保存到 ss[k*64] 中
            ss[i] = shape->GetStorageShape().GetDim(i);
            size[k] *= ss[i];
        }
        ndims[k] = shape->GetStorageShape().GetDimNum();
    }

    // if(dataType == ge::DT_INT32){
    //     return ge::GRAPH_FAILED;
    // }
    // if(size[1] != size[2]){
    //     return ge::GRAPH_FAILED;
    // }
    tiling.set_shape(shape);
    tiling.set_size(size);
    tiling.set_ndims(ndims);
    // if(size[0] != size[1]){
    //     printf("ScatterElements: The size of var and indices should be Equal");
    //     return ge::GRAPH_FAILED;
    // }
    uint32_t totalLength = size[1]; // 获取indices数据据总长度
    auto coreNum = ascendcPlatform.GetCoreNumAiv(); // 获取核心数量

    uint32_t pad32 = BLOCK_SIZE; // 设置填充大小为块大小

    if (totalLength > pad32 * coreNum) { 
        coreNum =
            totalLength % pad32 ? totalLength / pad32 + 1 : totalLength / pad32; // 计算核心数量
    }

    uint32_t tileNumMean = 0; // 初始化平均瓦片数量
    uint32_t tileNumEnd = 0; // 初始化结束瓦片数量
    uint32_t tileLengthMean = 0; // 初始化平均瓦片长度
    uint32_t tileLengthEnd = 0; // 初始化结束瓦片长度

    // 如果总数据比32B还小，直接当尾数处理
    if (totalLength < pad32) {
        tileNumMean = 1; // 平均瓦片数量为1
        tileNumEnd = 1; // 结束瓦片数量为1
        tileLengthMean = totalLength; // 平均瓦片长度为总长度
        tileLengthEnd = totalLength; // 结束瓦片长度为总长度
    } else {  // 总数据至少比 block_size 大时
        tileNumMean = totalLength / pad32;
        tileNumEnd = coreNum - tileNumMean;
        tileLengthMean = pad32;
        tileLengthEnd = totalLength % pad32;
    }

    tiling.set_totalLength(totalLength); // 设置平铺数据的总长度
    tiling.set_tileNumMean(tileNumMean); // 设置平均瓦片数量
    tiling.set_tileNumEnd(tileNumEnd); // 设置结束瓦片数量
    tiling.set_tileLengthMean(tileLengthMean); // 设置平均瓦片长度
    tiling.set_tileLengthEnd(tileLengthEnd); // 设置结束瓦片长度     

    // TODO: 4. 多核

    
    // 获得 axis 信息
    int32_t axis = *context->GetAttrs()->GetInt(0);
    if(axis<0){
        axis = numdims + axis;
    }
    tiling.set_axis(axis);

    // check dims and shape: https://github.com/pytorch/pytorch/blob/a8ac3a6b20b154e661e80ef930c6b0c563f3bb2d/aten/src/ATen/native/ScatterGatherChecks.h#L68
    // check dims and shape 
    //  1. index.size(d) <= self.size(d) for all d != dim
    //  2. index.size(d) <= src.size(d) for all d if src is a Tensor
    //  3. index.dim() == self.dim() == src.dim()
    if(size[2] != 1){
        if ((ndims[1] != ndims[0]) || (ndims[1] != ndims[2])) {
            printf("ScatterElements: The dims of indices, var and updates should be Equal");
            return ge::GRAPH_FAILED;
        }
        for (int32_t i = 0; i < ndims[1]; i++) {
            if(shape[i + 64] > shape[i]){
                printf("ScatterElements: index.size(d) <= src.size(d) for all d if src is a Tensor");
                return ge::GRAPH_FAILED;
            }
            if(axis == i){
                continue;
            }
            if (shape[i+64] > shape[i + 128]) {
                printf("ScatterElements: index.size(d) <= self.size(d) for all d != dim");
                return ge::GRAPH_FAILED;
            }
        }
    }
    // 获取 reduction 的值，并设置传入 kernel 的 mode 的值
    const char* reduction = context->GetAttrs()->GetStr(1);
    const char* mode1 = "assign";
    const char* mode2 = "add";
    const char* mode3 = "multiply";
    int32_t str_len = strlen(reduction);
    int32_t mode = 0;
    //  mode  =  0  assign,  mode  =  1  add,  mode  =  2  multiply
    if (reduction == nullptr){
        mode = 1;
    }

    if (str_len == strlen(mode1)){
        for (int32_t i = 0; i < str_len; i++){
            if (reduction[i] != mode1[i]){
                break;
            }
            if (i == str_len - 1){
                mode = 1;
            }
        }
    }

    if (str_len == strlen(mode2)){
        for (int32_t i = 0; i < str_len; i++){
            if (reduction[i] != mode2[i]){
                break;
            }
            if (i == str_len - 1){
                mode = 2;
            }
        }
    }

    if (str_len == strlen(mode3)){
        for (int32_t i = 0; i < str_len; i++){
            if (reduction[i] != mode3[i]){
                break;
            }
            if (i == str_len - 1){
                mode = 3;
            }
        }
    }
    tiling.set_mode(mode);

    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    currentWorkspace[0] = 0;// 设置总的workspace的数值大小，总的workspace空间由框架来申请并管理。
  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    // const gert::Shape* x1_shape = context->GetInputShape(0);
    // gert::Shape* y_shape = context->GetOutputShape(0);
    // *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class ScatterElements : public OpDef {
public:
    explicit ScatterElements(const char* name) : OpDef(name)
    {
        this->Input("var")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("updates")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("axis").AttrType(OPTIONAL).Int(0);
        this->Attr("reduce").AttrType(OPTIONAL).String("assign");

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(ScatterElements);
}
