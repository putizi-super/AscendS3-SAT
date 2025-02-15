
#include "non_max_suppression_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    NonMaxSuppressionTilingData tiling;

    const gert::StorageShape* shape_boxes_shape = context->GetInputShape(0);
    uint32_t num_batches = shape_boxes_shape->GetStorageShape().GetDim(0);
    uint32_t spatial_dimension = shape_boxes_shape->GetStorageShape().GetDim(1);

    const gert::StorageShape* shape_scores_shape = context->GetInputShape(1);
    uint32_t num_classes = shape_scores_shape->GetStorageShape().GetDim(1);

    const gert::StorageShape* shape_selected_indices = context->GetOutputShape(0);
    uint32_t num_selected_indices = shape_selected_indices->GetStorageShape().GetDim(0);

    const int64_t *center_point_box_ptr = (context->GetAttrs()->GetInt(0));
    int32_t center_point_box = *center_point_box_ptr;
    tiling.set_num_batches(num_batches);
    tiling.set_spatial_dimension(spatial_dimension);
    tiling.set_num_classes(num_classes);
    tiling.set_num_selected_indices(num_selected_indices);
    tiling.set_center_point_box(center_point_box);

    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    printf("log1");
    // const gert::Shape* shape_selected_indices = context->GetInputShape(5);
    // int64_t num_selected_indices = shape_selected_indices->GetDim(0);
    // gert::Shape* selected_indices = context->GetOutputShape(0);

    // selected_indices->SetDimNum(2);
    // selected_indices->SetDim(0, num_selected_indices);
    // selected_indices->SetDim(1, 3);
    return GRAPH_SUCCESS;
}
}


namespace ops {
class NonMaxSuppression : public OpDef {
public:
    explicit NonMaxSuppression(const char* name) : OpDef(name)
    {
        this->Input("boxes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("scores")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("max_output_boxes_per_class")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("iou_threshold")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("score_threshold")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("selected_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("center_point_box").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(NonMaxSuppression);
}
