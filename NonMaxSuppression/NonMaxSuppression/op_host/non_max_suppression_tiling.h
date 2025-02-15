
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(NonMaxSuppressionTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, num_batches);
  TILING_DATA_FIELD_DEF(uint32_t, spatial_dimension);
  TILING_DATA_FIELD_DEF(uint32_t, num_classes);
  TILING_DATA_FIELD_DEF(uint32_t, num_selected_indices);
  TILING_DATA_FIELD_DEF(int, center_point_box);
  
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NonMaxSuppression, NonMaxSuppressionTilingData)
}
