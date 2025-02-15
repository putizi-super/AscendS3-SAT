
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterElementsTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength); // 平铺总长度字段
  TILING_DATA_FIELD_DEF(uint32_t, tileNumMean); // 平铺数量均值字段
  TILING_DATA_FIELD_DEF(uint32_t, tileNumEnd); // 平铺数量结束字段
  TILING_DATA_FIELD_DEF(uint32_t, tileLengthMean); // 平铺长度均值字段
  TILING_DATA_FIELD_DEF(uint32_t, tileLengthEnd); // 平铺长度结束字段
  TILING_DATA_FIELD_DEF(int32_t, mode);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 192, shape); // 存储每个输入的shape信息
  TILING_DATA_FIELD_DEF_ARR(int32_t, 3, size); // 存储每个输入的元素个数信息
  TILING_DATA_FIELD_DEF_ARR(int32_t, 3, ndims); // 存储每个输入的 dims 数量
  TILING_DATA_FIELD_DEF(int32_t, axis);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterElements, ScatterElementsTilingData)
}
