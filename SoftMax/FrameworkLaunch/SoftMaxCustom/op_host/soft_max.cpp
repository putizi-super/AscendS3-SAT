
#include "soft_max_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
enum Condition {FIRST=0, MIDDLE, LAST};
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    SoftMaxTilingData tiling;
//   const gert::StorageShape* x1_shape = context->GetInputShape(0);
//   int32_t data_sz = 1;
//   for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
//     data_sz *= x1_shape->GetStorageShape().GetDim(i);
//   tiling.set_size(data_sz);
//   context->SetBlockDim(8);
//   tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
//   context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();

    // 这里获取data type是为了在算子init时做tile用，否则无法准确划分对应大小的buffer
    auto data_type = context->GetInputTensor(0)->GetDataType();
    // 获取输入数据一共有几个维度
    uint32_t dimNum = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    // 暂时没想好怎么把一个int传进来，暂时手动设定softmax维度
    const int* dimAttr = context->GetAttrs()->GetAttrPointer<int>(0);
    int dim = *dimAttr;
    // 确保传入的dim信息是正常的，不会产生越界
    // AscendC::ASSERT((dim >=0 && dim < dimNum) || (dim < 0 && dim >= -dimNum));
    // 处理dim传入为负数的索引情况，将负数索引转换成正常的正数索引
    dim = dim >= 0 ? dim : (dim + dimNum);
    // 根据dimnum获取最后一个维度的长度，该信息至关重要
    uint32_t lastDim =  context->GetInputShape(0)->GetStorageShape().GetDim(dimNum - 1);
    // const int* dim = context->GetAttrs()->GetAttrPointer<int>(3);

    // 为了处理任意shape的任意维度softmax，我们简单地分为3类情况进行讨论，但是需要注意的是，无论哪种情况，为了处理数据的向量化，数据都应该通过Padding对齐到block上
    /*
    1. 在最后一个维度上做softmax，那么此时前面不管有多少个维度，都可以融合成一个，使数据成为二维矩阵
        (a1, a2, a3,......, an, b) --> (a1a2a3......an, b)
        此时softmax的计算方式为默认的实现，先通过向量方法对每一行的b个数据进行reduce_sum，然后用向量方法对这一行的数据进行与1/reduce_sum的向量乘法（使用向量乘法而不是直接使用向量除法是因为乘法更快)

    2. 在第一个维度上做softmax，那么此时后面不管有多少个维度，都可以融合成一个，使数据成为二维矩阵
        (a, b1, b2, b3,......, bn) --> (a, b1b2b3......bn)
        此时softmax的计算方式应当发生改变，因为是对二维矩阵的每一列数据做reduce_sum，但是一列数据在内存中是不连续的，因此没有办法使用向量的reduce_sum方法
        但与此同时，reduce_tensor的长度本身就是b1b2b3......bn，因此可以用向量的加法循环将矩阵的每一行都加到reduce_tensor中(reduce_tensor需要提前置0)
        最终矩阵的每一行向量除reduce_tensor即可。
        需要注意的是向量方法都要求Local_tensor的起始地址按照block对齐，因此前面的数据padding是必要的

    3. 在不少于3个维度的数据上做某个中间维度的softmax，该维度之前的维度可以融合成一个维度，该维度之后的维度可以融合为一个维度，然后做数据padding
        (a1,...,an, b, c1,..., cn) --> (a1...an, b, c1...cn)
        此时的数据处理方式应和第2中方案一致，因为a1...an相当于batch_size
    */
   // 针对以上3种情况，我们需要进行不同的数据划分方式，以保持一个Block/tile的数据处理后就不用再进行计算
   /*
   1. 保证一行的数据在一个tile内就好
        tile0 --> | Block0 --> ****************######
        tile1 --> |            ****************######
        tile2 --> |            ****************######
        tile0 --> | Block1 --> ****************######
        tile1 --> |            ****************######
        tile2 --> |            ****************######
                               |<--raw_data-->|<pad>|
    2. 保证每一列的数据在一个tile内
                               |<--raw_data-->|<pad>|
                               ****************######
                               ****************######
                               ****************######
                               ****************######
                               ****************######
                               ****************######
                               ^         ^
                               |         |
                            Block0     Block1
    3. 从batch的维度进行划分

                        Block2 --> ****************######
                      Block1 --> ****************########
                    Block0 --> ****************##########
                               ****************##########
                               ****************##########
                               ****************##########
                               ****************########
                               ****************######
   
   */
    Condition conditionMod;
    uint32_t firstDim_length = 1;
    uint32_t lastDim_length = 1;
    uint32_t middleDim_length = context->GetInputShape(0)->GetStorageShape().GetDim(dim);
    if (dimNum >= 3 && dim > 0 && dim < (dimNum-1)) {
        conditionMod = MIDDLE;
        for(uint32_t i = 0; i < dim; i++){
            firstDim_length *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
        }
        for(uint32_t i = dim+1; i < dimNum; i++){
            lastDim_length *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
        }  
    } else if(dim == 0){
        conditionMod = FIRST;
        firstDim_length = middleDim_length = context->GetInputShape(0)->GetStorageShape().GetDim(0);
        for(uint32_t i = dim+1; i < dimNum; i++){
            lastDim_length *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
        }
    } else {
        conditionMod = LAST;
        for(uint32_t i = 0; i < dimNum-1; i++){
            firstDim_length *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
        }
        lastDim_length = context->GetInputShape(0)->GetStorageShape().GetDim(dimNum - 1);
    }
    
    tiling.set_totalLength(totalLength);
    tiling.set_dimNum(dimNum);
    tiling.set_firstDim(firstDim_length);
    tiling.set_middleDim(middleDim_length);
    tiling.set_lastDim(lastDim_length);
    tiling.set_dim(dim);
    // 这个tileNum后期需要根据实际的Buffer最大容量以及算子的计算-传输性能进行重新设计
    tiling.set_tileNum(4);
    tiling.set_condition(conditionMod);
    tiling.set_dtype(data_type);

    // 这个目前真没有这么多的AICore，后面重新设计一下。
    context->SetBlockDim(32);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class SoftMax : public OpDef {
public:
    explicit SoftMax(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("dim").AttrType(OPTIONAL).Int(-1);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(SoftMax);
}
