
#include "is_close_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

constexpr uint32_t DATA_SIZE_4 = 4;
constexpr uint32_t DATA_SIZE_2 = 2;
constexpr uint32_t DATA_SIZE_1 = 1;
constexpr uint32_t BLOCK_SIZE = 32; // VECIN、VECCALC 、VECOUT属于Unified Buffer 存储单元要求32B对齐
namespace optiling {
  static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    IsCloseTilingData tiling; // 分割方法相关参数
    // get attr   // TODO release need to 注释
    const float* atol = context->GetAttrs()->GetFloat(0);
    const float* rtol = context->GetAttrs()->GetFloat(1);
    const bool* equalNan = context->GetAttrs()->GetBool(2);
    tiling.set_atol(*atol);
    tiling.set_rtol(*rtol);
    tiling.set_equalNan(*equalNan);
    printf("[op_host] input attr: atol[%f] rtol[%f] equalNan[%d] attr_num[%d]", *atol, *rtol, *equalNan, context->GetAttrs()->GetAttrNum());
  
    uint64_t ubSize;
    uint32_t bufferNum = 16;
    auto ascendcPlatform =
    platform_ascendc::PlatformAscendC(context->GetPlatformInfo()); // 获取平台硬件信息
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize); // 获取指定硬件UB内存的大小如L1、L0_A、L0_B、L2、UB 
    uint32_t dataType = context->GetInputDesc(0)->GetDataType();
    uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();// 数据的总个数
    uint32_t coreNum = 1; // 使用到的核心的数量，后面代码会修改
    // uint32_t coreNum = ascendcPlatform.GetCoreNumAiv(); 
    // uint32_t coreNum =  platform_ascendc::PlatformAscendC(context->GetPlatformInfo()).GetCoreNum();

    printf("\n[op_host] first coreNum[%d] \n", coreNum);
    
    uint32_t dataSize = 0; // 数据类型的字节数量有float32 float16 int32  
    switch (dataType) {
    case ge::DT_FLOAT:
      dataSize = DATA_SIZE_4;
      break;
    case ge::DT_FLOAT16:
      dataSize = DATA_SIZE_2;
      break;
    case ge::DT_INT32:
      dataSize = DATA_SIZE_4;
      break;
    case ge::DT_UINT8 :
       dataSize = DATA_SIZE_1; 
       break;
    default:
      dataSize = DATA_SIZE_4;
      break;
    }

    uint32_t pad32 = BLOCK_SIZE; // 一个存储单元要求32B对齐
    uint32_t padMax = ubSize / bufferNum / dataSize; // 

    // if (totalLength < pad32 * coreNum) { // 一轮多核并发就可以处理完 设置下需要用到的最小核数  假设设置coreNum=8，只有很少的数据时  比如33B数据一轮8核没必要 只需要2核就可以
    //   coreNum = totalLength % pad32 ? totalLength / pad32 + 1 : totalLength / pad32;
    // }
    context->SetBlockDim(coreNum);
    tiling.set_totalLength(totalLength);

    // 计算一些最后一轮的分界点位置  
    uint32_t tileNumMean = 0; 
    uint32_t tileNumEnd = 0;
    uint32_t tileLengthMean = 0;
    uint32_t tileLengthEnd = 0;
    uint32_t blockLengthMean = 0;
    uint32_t blockLengthEnd = 0;
    // 如果总数据比32B还小，直接当尾数处理
    if (totalLength < pad32) {
      blockLengthMean = pad32;
      blockLengthEnd = totalLength;
      tileNumMean = 1;
      tileNumEnd = 1;
      tileLengthMean = totalLength;
      tileLengthEnd = totalLength;
    }
    else {  // 总数据至少比32B大时
      // 总数据至少比32B大时
      uint32_t realTotalLength =
        totalLength % (pad32 * coreNum)
        ?  // 补足totalLength到32B倍核心数的整数倍
        ((totalLength / (pad32 * coreNum)) + 1) * (pad32 * coreNum)
        : totalLength;
      uint32_t maxBlockLength = realTotalLength / coreNum;
      if (realTotalLength - totalLength > maxBlockLength) {
        maxBlockLength = totalLength / coreNum;
      }

      if (maxBlockLength >
        padMax) {  // maxBlockLength大于padMax时对maxBlockLength进行判定
        uint32_t padTemp = 0;
        for (uint32_t i = padMax / 2; i <= padMax; i += pad32) {
          padTemp = maxBlockLength % i == 0 ? i : padTemp;
        }
        if (padTemp) {  // 如果maxBlockLength可以被PadTemp整除，那么padTemp就是tilelength
          blockLengthMean = maxBlockLength;
          blockLengthEnd = totalLength - blockLengthMean * (coreNum - 1);
          tileNumMean = blockLengthMean / padTemp;
          tileNumEnd = tileNumMean;
          tileLengthMean = padTemp;
          tileLengthEnd = blockLengthEnd - padTemp * (tileNumEnd - 1);
        }
        else {  // 如果maxBlockLength不能被PadTemp整除，那么padMax就是tilelength
          blockLengthMean = maxBlockLength - maxBlockLength % padMax;
          blockLengthEnd = totalLength - blockLengthMean * (coreNum - 1);
          tileNumMean = blockLengthMean / padMax;
          tileNumEnd = blockLengthEnd % padMax
            ? blockLengthEnd / padMax + 1
            : (blockLengthEnd /
              padMax);  // 计算最后一个核心会不会多一个尾数块
          if (padMax >= blockLengthEnd) {
            tileNumEnd = 1;
          }
          tileLengthMean = padMax;
          tileLengthEnd =
            blockLengthEnd -
            padMax * (tileNumEnd - 1);  // 计算最后一个核心的尾数块长度
        }
      }
      else {  // maxBlockLength小于padMax时直接取maxBlockLength中的最大Pad32倍数
        if (maxBlockLength >= pad32) {  // maxBlockLength大于pad32时
          blockLengthMean = maxBlockLength - maxBlockLength % pad32;
          blockLengthEnd = totalLength - blockLengthMean * (coreNum - 1);
          tileNumMean = 1;  // 只有一个tileNum
          tileNumEnd =
            blockLengthEnd % pad32
            ? blockLengthEnd / blockLengthMean + 1
            : blockLengthEnd /
            blockLengthMean;  // 如果尾块不能32B对齐则多分配一个尾块
          if (blockLengthMean >= blockLengthEnd) {
            tileNumEnd = 1;
          }
          tileLengthMean = blockLengthMean;
          tileLengthEnd =
            blockLengthEnd -
            tileLengthMean *
            (tileNumEnd - 1);  // 将尾数彻底分给最后一个核心的最后一个tile
        }
        else {  // maxBlockLength小于pad32时，前面的block优先分配32B数据
          blockLengthMean = pad32;
          blockLengthEnd = totalLength - pad32 * (coreNum - 1);
          tileNumMean = 1;
          tileNumEnd = 1;
          tileLengthMean = pad32;
          tileLengthEnd = blockLengthEnd;
        }
      }
    }
    printf("\n[op_host] final coreNum[%d] \n", coreNum);
    printf("\n[op_host] total_length[%d], tile_num_mean[%d],tile_num_end[%d], tile_length_mean[%d], tile_length_end[%d], block_length_mean[%d], block_length_end[%d] \n",totalLength, tileNumMean,tileNumEnd, tileLengthMean, tileLengthEnd, blockLengthMean, blockLengthEnd);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNumMean(tileNumMean);
    tiling.set_tileNumEnd(tileNumEnd);
    tiling.set_tileLengthMean(tileLengthMean);
    tiling.set_tileLengthEnd(tileLengthEnd);
    tiling.set_blockLengthMean(blockLengthMean);
    tiling.set_blockLengthEnd(blockLengthEnd);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
      context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
  }
}  // namespace optiling

namespace ge {
  static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
  }  


}// namespace ge

namespace ops {
  class IsClose : public OpDef {
    public:
      explicit IsClose(const char* name) : OpDef(name) {
        this->Input("x1")
          .ParamType(REQUIRED)
          .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_UINT8 , ge::DT_INT32 })
          .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
          .UnknownShapeFormat(
            { ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
        this->Input("x2")
          .ParamType(REQUIRED)
          .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_UINT8 , ge::DT_INT32 })
          .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
          .UnknownShapeFormat(
            { ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
        this->Output("y")
          .ParamType(REQUIRED)
          .DataType({ ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL })
          .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
          .UnknownShapeFormat(
            { ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
        this->Attr("atol").AttrType(OPTIONAL).Float(1e-08);
        this->Attr("rtol").AttrType(OPTIONAL).Float(1e-05);
    
        this->Attr("equal_nan").AttrType(OPTIONAL).Bool(false);
        this->SetInferShape(ge::InferShape);

        this->AICore().SetTiling(optiling::TilingFunc);

        this->AICore().AddConfig("ascend310b");
      }
  };
  OP_ADD(IsClose);
} // namespace ops
