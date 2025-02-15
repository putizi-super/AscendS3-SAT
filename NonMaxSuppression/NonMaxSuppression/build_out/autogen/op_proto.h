#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(NonMaxSuppression)
    .INPUT(boxes, ge::TensorType::ALL())
    .INPUT(scores, ge::TensorType::ALL())
    .INPUT(max_output_boxes_per_class, ge::TensorType::ALL())
    .INPUT(iou_threshold, ge::TensorType::ALL())
    .INPUT(score_threshold, ge::TensorType::ALL())
    .OUTPUT(selected_indices, ge::TensorType::ALL())
    .ATTR(center_point_box, Int, 0)
    .OP_END_FACTORY_REG(NonMaxSuppression);

}

#endif
