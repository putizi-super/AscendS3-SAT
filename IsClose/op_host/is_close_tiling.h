
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IsCloseTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength); // 数据总长
  TILING_DATA_FIELD_DEF(uint32_t, tileNumMean); // 
  TILING_DATA_FIELD_DEF(uint32_t, tileNumEnd);
  TILING_DATA_FIELD_DEF(uint32_t, tileLengthMean);
  TILING_DATA_FIELD_DEF(uint32_t, tileLengthEnd);
  TILING_DATA_FIELD_DEF(uint32_t, blockLengthMean);
  TILING_DATA_FIELD_DEF(uint32_t, blockLengthEnd);

  TILING_DATA_FIELD_DEF(float, atol);
  TILING_DATA_FIELD_DEF(float, rtol);
  TILING_DATA_FIELD_DEF(bool, equalNan);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IsClose, IsCloseTilingData)
}
