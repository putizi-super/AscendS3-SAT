#include <cstddef> // 引入cstddef头文件，定义了size_t等类型
#include <cstdint> // 引入cstdint头文件，定义了固定宽度整数类型
#include <cstring> // 引入cstring头文件，提供C风格字符串处理功能

#include "not_equal_tiling.h" // 引入not_equal_tiling.h头文件，定义平铺数据结构
#include "register/op_def_registry.h" // 引入op_def_registry.h头文件，注册操作定义
#include "tiling/platform/platform_ascendc.h" // 引入平台相关头文件

constexpr uint32_t DATA_SIZE_4 = 4; // 定义4字节数据大小
constexpr uint32_t DATA_SIZE_2 = 2; // 定义2字节数据大小
constexpr uint32_t DATA_SIZE_1 = 1; // 定义1字节数据大小
constexpr uint32_t BLOCK_SIZE = 32; // 定义块大小为32

constexpr int32_t inputVarNum = 2; // 输入个数
constexpr int32_t maxDimNum = 64; // 最大维度数量

namespace optiling { // 定义命名空间optiling
  static ge::graphStatus TilingFunc(gert::TilingContext* context) { // 定义平铺函数
    TilingData tiling; // 创建TilingData实例
    uint64_t ubSize; // 定义ubSize变量
    uint32_t bufferNum = 16; // 定义16个buffer
    auto ascendcPlatform =platform_ascendc::PlatformAscendC(context->GetPlatformInfo()); // 获取AscendC平台信息
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize); // 获取核心内存大小
    uint32_t dataType = context->GetInputDesc(0)->GetDataType(); // 获取输入数据类型
    
    // 初始化变量
    uint32_t x1_length = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t x2_length = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
    
    // 此处假定 dim 维度相等
    int64_t DimNum1 = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    int64_t DimNum2 = context->GetInputShape(1)->GetStorageShape().GetDimNum();
    int64_t DimNum = DimNum1;
    int64_t shape[maxDimNum * inputVarNum], shapefull[maxDimNum];

    // 获得每个输入的shape
    for (int k = 0; k < inputVarNum; ++k) {
        int64_t *ss = &shape[k * maxDimNum];
        const gert::StorageShape* inputshape = context->GetInputShape(k);
        for (int i = 0; i < inputshape->GetStorageShape().GetDimNum(); i++) {
            ss[i] = inputshape->GetStorageShape().GetDim(i);
        }
    }
    // 获得 广播后的 shape, 以及总长度
    uint32_t totalLength = 1;
    for (int k = 0; k < DimNum; ++k) {
        int64_t *ss = &shape[0];
        int64_t *sf = &shapefull[0];
        sf[k] = (ss[k] > ss[k + maxDimNum]) ? ss[k] : ss[k + maxDimNum];   
        totalLength *= sf[k];
    }

    tiling.set_DimNum(DimNum);
    tiling.set_shape(shape);
    tiling.set_shapefull(shapefull);

    // tiling
    auto coreNum = ascendcPlatform.GetCoreNumAiv(); // 获取核心数量
    uint32_t dataSize = 0; // 初始化数据大小
    switch (dataType) { // 根据输入数据类型获取数据大小
      case ge::DT_FLOAT:
        dataSize = DATA_SIZE_4; // 浮点型数据大小为4
        break;
      case ge::DT_FLOAT16:
        dataSize = DATA_SIZE_2; // 浮点16型数据大小为2
        break;
      case ge::DT_INT8:
        dataSize = DATA_SIZE_1; // 整型8型数据大小为1
        break;
      case ge::DT_INT32:
        dataSize = DATA_SIZE_4; // 整型32型数据大小为4
        break;
      default:
        dataSize = DATA_SIZE_4; // 默认数据大小为4
        break;
    }

    uint32_t tileNumMean = 0; // 初始化平均瓦片数量
    uint32_t tileNumEnd = 0; // 初始化结束瓦片数量
    uint32_t tileLengthMean = 0; // 初始化平均瓦片长度
    uint32_t tileLengthEnd = 0; // 初始化结束瓦片长度
    uint32_t blockLengthMean = 0; // 初始化平均块长度
    uint32_t blockLengthEnd = 0; // 初始化结束块长度

    uint32_t ALIGN_NUM = 0; // 一个块需要对齐的数量
    uint32_t block_size = 0; 
    uint32_t core_size = 0;
    uint32_t core_remain = 0;

    uint32_t padMax = (ubSize / bufferNum / dataSize) / (2 * BLOCK_SIZE) * (2 * BLOCK_SIZE); // 计算最大填充大小  =  每个 buffer 中可以放的数据 最大数量
    //更新 coreNum
    if (totalLength < BLOCK_SIZE * coreNum) { // 如果总长度小于填充大小乘以核心数量
      coreNum =
          totalLength % BLOCK_SIZE ? totalLength / BLOCK_SIZE + 1 : totalLength / BLOCK_SIZE; // 计算核心数量
    }

    // 如果总数据比32B还小，直接当尾数处理
    if (totalLength < BLOCK_SIZE) {
      blockLengthMean = BLOCK_SIZE; // 块长度均值设置为填充大小
      blockLengthEnd = totalLength; // 结束块长度设置为总长度
      tileNumMean = 1; // 平均瓦片数量为1
      tileNumEnd = 1; // 结束瓦片数量为1
      tileLengthMean = totalLength; // 平均瓦片长度为总长度
      tileLengthEnd = totalLength; // 结束瓦片长度为总长度
    } else {  
      // 计算填充后 实际的 TotalLength, (单位是 ： coreNum * BLOCK_SIZE)
      uint32_t realTotalLength = totalLength % (BLOCK_SIZE * coreNum) ?  
              ((totalLength / (BLOCK_SIZE * coreNum)) + 1) * (BLOCK_SIZE * coreNum)
              : totalLength;
      
      if (coreNum == 0) {
        return ge::GRAPH_FAILED; // 如果核心数量为0，返回失败
      }

      uint32_t maxBlockLength = realTotalLength / coreNum; // 计算最大块长度
      if (realTotalLength - totalLength > maxBlockLength) {
        maxBlockLength = totalLength / coreNum; // 更新最大块长度
      }
    
      if (maxBlockLength > padMax) {  // maxBlockLength大于padMax时对maxBlockLength进行判定
        uint32_t padTemp = 0; // 初始化临时填充变量
        for (uint32_t i = padMax / 2; i <= padMax; i += BLOCK_SIZE) {
          padTemp = maxBlockLength % i == 0 ? i : padTemp; // 找到可被maxBlockLength整除的填充大小
        }
        if (padTemp) {  // 如果maxBlockLength可以被PadTemp整除，那么padTemp就是tilelength
          blockLengthMean = maxBlockLength; // 设置块长度均值为最大块长度
          blockLengthEnd = totalLength - blockLengthMean * (coreNum - 1); // 计算结束块长度
          tileNumMean = blockLengthMean / padTemp; // 计算平均瓦片数量
          tileNumEnd = tileNumMean; // 结束瓦片数量与平均瓦片数量相同
          tileLengthMean = padTemp; // 平均瓦片长度为填充大小
          tileLengthEnd = blockLengthEnd - padTemp * (tileNumEnd - 1); // 计算结束瓦片长度
        } else {  // 如果maxBlockLength不能被PadTemp整除，那么padMax就是tilelength
          blockLengthMean = maxBlockLength - maxBlockLength % padMax; // 设置块长度均值
          blockLengthEnd = totalLength - blockLengthMean * (coreNum - 1); // 计算结束块长度
          tileNumMean = blockLengthMean / padMax; // 计算平均瓦片数量
          tileNumEnd = blockLengthEnd % padMax
                          ? blockLengthEnd / padMax + 1
                          : (blockLengthEnd /
                              padMax);  // 计算最后一个核心会不会多一个尾数块
          if (padMax >= blockLengthEnd) {
            tileNumEnd = 1; // 如果padMax大于等于结束块长度，设置结束瓦片数量为1
          }
          tileLengthMean = padMax; // 平均瓦片长度为padMax
          tileLengthEnd =
              blockLengthEnd -
              padMax * (tileNumEnd - 1);  // 计算最后一个核心的尾数块长度
        }
      } else {  // maxBlockLength小于padMax时直接取maxBlockLength中的最大Pad32倍数
          if (maxBlockLength >= BLOCK_SIZE) {  // maxBlockLength大于pad32时
            blockLengthMean = maxBlockLength - maxBlockLength % BLOCK_SIZE; // 设置块长度均值
            blockLengthEnd = totalLength - blockLengthMean * (coreNum - 1); // 计算结束块长度
            tileNumMean = 1;  // 只有一个tileNum
            tileNumEnd =
                blockLengthEnd % BLOCK_SIZE
                    ? blockLengthEnd / blockLengthMean + 1
                    : blockLengthEnd /
                          blockLengthMean;  // 如果尾块不能32B对齐则多分配一个尾块
            if (blockLengthMean >= blockLengthEnd) {
              tileNumEnd = 1; // 如果块长度均值大于等于结束块长度，设置结束瓦片数量为1
            }
            tileLengthMean = blockLengthMean; // 平均瓦片长度为块长度均值
            tileLengthEnd =
                blockLengthEnd -
                tileLengthMean *
                    (tileNumEnd - 1);  // 将尾数彻底分给最后一个核心的最后一个tile
          } else {  // maxBlockLength小于pad32时，前面的block优先分配32B数据
            blockLengthMean = BLOCK_SIZE; // 块长度均值设置为填充大小
            blockLengthEnd = totalLength - BLOCK_SIZE * (coreNum - 1); // 计算结束块长度
            tileNumMean = 1; // 平均瓦片数量为1
            tileNumEnd = 1; // 结束瓦片数量为1
            tileLengthMean = BLOCK_SIZE; // 平均瓦片长度为填充大小
            tileLengthEnd = blockLengthEnd; // 结束瓦片长度为结束块长度
          }
      }
    }
    
    tiling.set_x1_length(x1_length);
    tiling.set_x2_length(x2_length);
    tiling.set_totalLength(totalLength); // 设置数据的总长度
    tiling.set_tileNumMean(tileNumMean); // 设置平均瓦片数量
    tiling.set_tileNumEnd(tileNumEnd); // 设置结束瓦片数量
    tiling.set_tileLengthMean(tileLengthMean); // 设置平均瓦片长度
    tiling.set_tileLengthEnd(tileLengthEnd); // 设置结束瓦片长度
    tiling.set_blockLengthMean(blockLengthMean); // 设置平均块长度
    tiling.set_blockLengthEnd(blockLengthEnd); // 设置结束块长度
    context->SetBlockDim(coreNum); 
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), // 保存平铺数据到缓冲区
                        context->GetRawTilingData()->GetCapacity()); // 获取缓冲区容量
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize()); // 设置缓冲区数据大小
    size_t* currentWorkspace = context->GetWorkspaceSizes(1); // 获取当前工作区大小
    currentWorkspace[0] = 0; // 初始化当前工作区大小为0
    return ge::GRAPH_SUCCESS; // 返回成功状态

  }  // TilingFunc
}// namespace optiling

namespace ge { // 定义命名空间ge
static ge::graphStatus InferShape(gert::InferShapeContext* context) { // 定义推断形状函数
  const gert::Shape* x1_shape = context->GetInputShape(0); // 获取输入形状
  gert::Shape* y_shape = context->GetOutputShape(0); // 获取输出形状
  *y_shape = *x1_shape; // 输出形状与输入形状相同
  return GRAPH_SUCCESS; // 返回成功状态
}
}  // namespace ge

namespace ops { // 定义命名空间ops
  class NotEqual : public OpDef { // 定义NotEqual类，继承自OpDef
  public:
    explicit NotEqual(const char* name) : OpDef(name) { // 构造函数
      this->Input("x1") // 定义输入x1
          .ParamType(REQUIRED) // 设置为必需参数
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32}) // 支持的数据类型
          .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}) // 支持的格式
          .UnknownShapeFormat(
              {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}); // 未知形状格式
      this->Input("x2") // 定义输入x2
          .ParamType(REQUIRED) // 设置为必需参数
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32}) // 支持的数据类型
          .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}) // 支持的格式
          .UnknownShapeFormat(
              {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}); // 未知形状格式
      this->Output("y") // 定义输出y
          .ParamType(REQUIRED) // 设置为必需参数
          .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL}) // 支持的数据类型
          .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}) // 支持的格式
          .UnknownShapeFormat(
              {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}); // 未知形状格式

      this->SetInferShape(ge::InferShape); // 设置推断形状函数

      this->AICore().SetTiling(optiling::TilingFunc); // 设置平铺函数

      this->AICore().AddConfig("ascend310p")
                    .AddConfig("ascend310b");
    }
  };

  OP_ADD(NotEqual);
}  // namespace ops
