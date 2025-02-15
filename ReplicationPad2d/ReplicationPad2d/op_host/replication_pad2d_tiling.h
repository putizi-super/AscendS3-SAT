
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ReplicationPad2dTilingData)

  TILING_DATA_FIELD_DEF(int32_t, padL); 
  TILING_DATA_FIELD_DEF(int32_t, padR); 
  TILING_DATA_FIELD_DEF(int32_t, padT); 
  TILING_DATA_FIELD_DEF(int32_t, padB); 

  TILING_DATA_FIELD_DEF(uint32_t, blocksize); 
  TILING_DATA_FIELD_DEF(int32_t, XDim);
  TILING_DATA_FIELD_DEF(int32_t, YDim);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 10, Xshape);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 10, Yshape);
  TILING_DATA_FIELD_DEF(uint32_t, blockLengthMean);
  TILING_DATA_FIELD_DEF(uint32_t, blockLengthEnd);
  TILING_DATA_FIELD_DEF(int32_t, totalSizeX);
  TILING_DATA_FIELD_DEF(int32_t, totalSizeY);

  TILING_DATA_FIELD_DEF(uint32_t, lastDim);
  TILING_DATA_FIELD_DEF(uint32_t, lastDimY);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ReplicationPad2d, ReplicationPad2dTilingData)
}
