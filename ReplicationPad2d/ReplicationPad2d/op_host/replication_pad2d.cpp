
#include "replication_pad2d_tiling.h"
#include "register/op_def_registry.h"

constexpr uint32_t DATA_SIZE_1 = 4; // 定义4字节数据大小 fp32
constexpr uint32_t DATA_SIZE_2 = 2; // 定义2字节数据大小 fp16
constexpr uint32_t BLOCK_SIZE = 32; // 定义块大小 为 32B
constexpr int32_t maxDimNum = 10; // 最大维度数量
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    ReplicationPad2dTilingData tiling;
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    const gert::StorageShape* y_shape = context->GetOutputShape(0);

    // 获得 x 的 shape、dim 信息
    int32_t Xshape[maxDimNum];
    int32_t totalSizeX = 1;
    for (int i = 0; i < x_shape->GetStorageShape().GetDimNum(); i++) {
        Xshape[i] = x_shape->GetStorageShape().GetDim(i);   
        totalSizeX *= Xshape[i];     
    }
    int32_t XDim = x_shape->GetStorageShape().GetDimNum();
    if(XDim < 3 || XDim > 4) {
        return ge::GRAPH_FAILED;
    }
    
    // 获得 y dim 以及 shape 信息
    int32_t YDim = XDim;
    int32_t Yshape[maxDimNum];
    int32_t totalSizeY = 1;
    for (int i = 0; i < YDim; i++) {
        Yshape[i] = y_shape->GetStorageShape().GetDim(i);    
        totalSizeY *= Yshape[i];     
    }


    // 根据数据类型设置 块大小
    uint32_t dataType = context->GetInputDesc(0)->GetDataType(); // 获取输入数据类型
    uint32_t dataSize = 0; // 初始化数据大小
    switch (dataType) { // 根据输入数据类型获取数据大小
        case ge::DT_FLOAT:
            dataSize = DATA_SIZE_1; // 浮点型数据大小为4
            break;
        case ge::DT_FLOAT16:
            dataSize = DATA_SIZE_2; // 浮点16型数据大小为2
            break;
        default:
            dataSize = DATA_SIZE_1; // 默认数据大小为4
            break;
    }
    uint32_t block_size = BLOCK_SIZE / dataSize; 
    uint32_t blockLengthMean = 0; // 平均块长度
    uint32_t blockLengthEnd = 0;  // 结束块长度
    uint32_t lastDim = Xshape[XDim - 1];
    uint32_t lastDimY = Yshape[YDim - 1];
    if (lastDim % block_size == 0) {
        blockLengthMean = block_size;
        blockLengthEnd = 0;
    } else {
        if(lastDim < block_size){
            blockLengthMean = 0;
            blockLengthEnd = lastDim;
        }else{
            blockLengthMean = block_size;
            blockLengthEnd = lastDim % block_size;
        }
    }
    
    // 获得 pad相关数值
    const gert::Tensor *pad_tensor = context->GetInputTensor(1); // 获取第1个输入的tensor
    auto pad_addr = pad_tensor->GetData<int32_t>(); 
    int32_t padding_left = pad_addr[0];
    int32_t padding_right = pad_addr[1];
    int32_t padding_top = pad_addr[2];
    int32_t padding_bottom = pad_addr[3]; 


    tiling.set_XDim(XDim);
    tiling.set_YDim(YDim);
    tiling.set_blocksize(block_size);
    tiling.set_Xshape(Xshape);
    tiling.set_Yshape(Yshape);
    tiling.set_blockLengthMean(blockLengthMean);
    tiling.set_blockLengthEnd(blockLengthEnd);

    tiling.set_lastDim(lastDim);
    tiling.set_lastDimY(lastDimY);
    tiling.set_padL(padding_left);
    tiling.set_padR(padding_right);
    tiling.set_padT(padding_top);
    tiling.set_padB(padding_bottom);

    tiling.set_totalSizeX(totalSizeX);
    tiling.set_totalSizeY(totalSizeY);
    context->SetBlockDim(1); // 使用一个核

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    const gert::Tensor *pad_tensor = context->GetInputTensor(1); // 获取第1个输入的tensor
    gert::Shape* y_shape = context->GetOutputShape(0);
    // shape 推断
    if (x_shape == nullptr || pad_tensor == nullptr || y_shape == nullptr) {
        // 防御式编程，不应该出现的场景，打印错误并返回失败
        return ge::GRAPH_FAILED;
    }
    auto pad_size = static_cast<int32_t>(pad_tensor->GetShapeSize());
    if (pad_size !=  4) {
        // 防御式编程，不应该出现的场景，打印错误并返回失败
        return ge::GRAPH_FAILED;
    }
    auto addr = pad_tensor->GetData<int32_t>();
    int32_t padding_left = addr[0];
    int32_t padding_right = addr[1];
    int32_t padding_top = addr[2];
    int32_t padding_bottom = addr[3];
    
    // 设置 DimNum
    int32_t x_DimNum = x_shape->GetDimNum();
    y_shape->SetDimNum(x_DimNum);

    // 设置 shape
    y_shape->SetDim(x_DimNum - 2, x_shape->GetDim(x_DimNum - 2) + padding_top + padding_bottom);
    y_shape->SetDim(x_DimNum - 1, x_shape->GetDim(x_DimNum - 1) + padding_left + padding_right);

    return GRAPH_SUCCESS;
}
}


namespace ops {
class ReplicationPad2d : public OpDef {
public:
    explicit ReplicationPad2d(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("paddings")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(ReplicationPad2d);
}
