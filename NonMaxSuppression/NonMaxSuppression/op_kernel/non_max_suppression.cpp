#include "kernel_operator.h"

template<typename T>
class NonMaxSuppressionKernel {
public:
    __aicore__ inline NonMaxSuppressionKernel() {}
    __aicore__ inline void Init(GM_ADDR boxes, GM_ADDR scores, GM_ADDR max_output_boxes_per_class, GM_ADDR iou_threshold, GM_ADDR score_threshold, GM_ADDR selected_indices,
    uint32_t num_batches, uint32_t spatial_dimension, 
    uint32_t num_classes, uint32_t num_selected_indices,
    int center_point_box) {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        this->num_batches = num_batches;
        this->spatial_dimension = spatial_dimension;
        this->num_classes = num_classes;    
        this->num_selected_indices = num_selected_indices;
        this->center_point_box = center_point_box;

        this->boxesGm.SetGlobalBuffer((__gm__ float*)boxes, num_batches * spatial_dimension * 4);
        this->scoresGm.SetGlobalBuffer((__gm__ float*)scores, num_batches * num_classes * spatial_dimension);
        this->max_output_boxes_per_classGm.SetGlobalBuffer((__gm__ int32_t*)max_output_boxes_per_class, 1);
        this->iou_thresholdGm.SetGlobalBuffer((__gm__ float*)iou_threshold, 1);
        this->score_thresholdGm.SetGlobalBuffer((__gm__ float*)score_threshold, 1);
        this->selected_indicesGm.SetGlobalBuffer((__gm__ int32_t*)selected_indices, num_selected_indices * 3);

        this->max_output_boxes_per_class = this->max_output_boxes_per_classGm.GetValue(0); // int32_t
        this->iou_threshold = this->iou_thresholdGm.GetValue(0); // float
        this->score_threshold = this->score_thresholdGm.GetValue(0); // float
        AscendC::printf("max_output_boxes_per_class: %d\n", this->max_output_boxes_per_class);
        AscendC::printf("iou_threshold: %f\n", this->iou_threshold);
        AscendC::printf("score_threshold: %f\n", this->score_threshold);
        AscendC::printf("num_batches: %d\n", this->num_batches);
        AscendC::printf("num_classes: %d\n", this->num_classes);
        AscendC::printf("spatial_dimension: %d\n", this->spatial_dimension);
        // AscendC::printf()
    }
    __aicore__ inline void Process() {
        int32_t candidateIndices[1024];
        uint32_t selectedIndicesOffset = 0;
        // 遍历所有 batch
        for (int32_t batchIdx = 0; batchIdx < num_batches; ++batchIdx) {
            // 加载当前 batch 的 boxes 和 scores 对应的偏移量
            uint32_t boxesOffset = batchIdx * spatial_dimension * 4;
            uint32_t scoresOffset = batchIdx * num_classes * spatial_dimension; // 当前框在所有 类别中的置信度

            // 遍历所有类别
            for (int32_t classIdx = 0; classIdx < num_classes; ++classIdx) { // 获得当前类所有框的置信度
                // 获得当前类别的 scores 的偏移量
                uint32_t curScoresOffset = scoresOffset + classIdx * spatial_dimension;
                uint32_t curBoxesOffset = boxesOffset;
                // 筛选出分数大于 scoreThreshold 的边界框
                int32_t candidateCount = 0;

                for (int32_t boxIdx = 0; boxIdx < spatial_dimension; ++boxIdx) {
                    float score = scoresGm.GetValue(curScoresOffset + boxIdx);
                    if (score >= score_threshold) {
                        candidateIndices[candidateCount++] = boxIdx;
                    }
                }

                // 按分数从高到低排序
                for (int32_t i = 0; i < candidateCount - 1; ++i) {
                    for (int32_t j = i + 1; j < candidateCount; ++j) {
                        float scoreI = scoresGm.GetValue(curScoresOffset + candidateIndices[i]);
                        float scoreJ = scoresGm.GetValue(curScoresOffset + candidateIndices[j]);
                        if (scoreI < scoreJ) {
                            int32_t temp = candidateIndices[i];
                            candidateIndices[i] = candidateIndices[j];
                            candidateIndices[j] = temp;
                        }
                    }
                }

                // 非极大值抑制
                int32_t selectedIndices[1024];
                int32_t selectedCount = 0;
                int8_t flag_keep[1024] = {0};

                for (int32_t i = 0; i < candidateCount; ++i) {
                    if(flag_keep[i] == 1) continue; // 如果被抑制, 就跳过
                    int32_t idx1 = candidateIndices[i];
                    selectedIndices[selectedCount++] = idx1;
                    flag_keep[i] = 1; // 如果选中了,就抑制掉
                    
                    if(selectedCount > max_output_boxes_per_class) break;
                    
                    // AscendC::printf("batch_id:%d , class_id:%d, selectedidx:%d\n", batchIdx, classIdx, idx1);
                    // 抑制其他的
                    for (int32_t j = 0; j < candidateCount; ++j) {

                        if(flag_keep[j] == 1) continue;
                        
                        int32_t idx2 = candidateIndices[j];
                        // 计算 IoU
                        float iou = this->calculateIoU(curBoxesOffset + idx1 * 4, 
                                            curBoxesOffset + idx2 * 4, this->center_point_box);
                        if (iou >= iou_threshold) {
                            flag_keep[j] = 1; // 抑制
                        }
                    }
                }
                // 将选中的边界框索引存储到 selected_indices 中
                for (int32_t i = 0; i < max_output_boxes_per_class; ++i) {
                    int32_t idx = selectedIndices[i];
                    // AscendC::PRINTF("selectedIndicesOffset: %d, batchIdx: %d, classIdx: %d, idx: %d\n",selectedIndicesOffset, batchIdx, classIdx, idx);
                    selected_indicesGm.SetValue(selectedIndicesOffset, batchIdx);
                    selected_indicesGm.SetValue(selectedIndicesOffset + 1, classIdx);
                    selected_indicesGm.SetValue(selectedIndicesOffset + 2, idx);
                    selectedIndicesOffset += 3;
                }
            }
        }
    }

private:

    __aicore__ inline float calculateIoU(uint32_t box1_offset, uint32_t box2_offset, int box_format = 0) {
        if (box_format == 1) {
            // 中心模式
            return this->calculateIoUCenterMode(box1_offset, box2_offset);
        } else {
            // 角点模式
            return this->calculateIoUCornerMode(box1_offset, box2_offset);
        }
    }

    __aicore__ inline float calculateIoUCornerMode(uint32_t box1_offset, uint32_t box2_offset) {
        // 获取边界框的角点坐标
        float y1_1 = boxesGm.GetValue(box1_offset);      // y1 of box1
        float x1_1 = boxesGm.GetValue(box1_offset + 1);  // x1 of box1
        float y2_1 = boxesGm.GetValue(box1_offset + 2);  // y2 of box1
        float x2_1 = boxesGm.GetValue(box1_offset + 3);  // x2 of box1

        float y1_2 = boxesGm.GetValue(box2_offset);      // y1 of box2
        float x1_2 = boxesGm.GetValue(box2_offset + 1);  // x1 of box2
        float y2_2 = boxesGm.GetValue(box2_offset + 2);  // y2 of box2
        float x2_2 = boxesGm.GetValue(box2_offset + 3);  // x2 of box2

        // 确保 x1 和 x2 是对角线的两个角点
        float box1_xmin = min(x1_1, x2_1);
        float box1_xmax = max(x1_1, x2_1);
        float box1_ymin = min(y1_1, y2_1);
        float box1_ymax = max(y1_1, y2_1);

        float box2_xmin = min(x1_2, x2_2);
        float box2_xmax = max(x1_2, x2_2);
        float box2_ymin = min(y1_2, y2_2);
        float box2_ymax = max(y1_2, y2_2);

        // 计算交集的左上角和右下角坐标
        float x1 = max(box1_xmin, box2_xmin);  // 交集的左边界
        float y1 = max(box1_ymin, box2_ymin);  // 交集的上边界
        float x2 = min(box1_xmax, box2_xmax);  // 交集的右边界
        float y2 = min(box1_ymax, box2_ymax);  // 交集的下边界

        // 计算交集的宽度和高度
        float w = max(static_cast<float>(0), x2 - x1);  // 交集的宽度
        float h = max(static_cast<float>(0), y2 - y1);  // 交集的高度
        float inter = w * h;  // 交集面积

        // 计算两个边界框的面积
        float area1 = (box1_xmax - box1_xmin) * (box1_ymax - box1_ymin);  // box1 的面积
        float area2 = (box2_xmax - box2_xmin) * (box2_ymax - box2_ymin);  // box2 的面积

        // 计算并集面积
        float union_area = area1 + area2 - inter;

        // 计算 IoU
        return inter / union_area;
    }

    __aicore__ inline float calculateIoUCenterMode(uint32_t box1_offset, uint32_t box2_offset) {
        // 获取边界框的中心点坐标和宽高
        float cx1 = boxesGm.GetValue(box1_offset);      // x_center of box1
        float cy1 = boxesGm.GetValue(box1_offset + 1);  // y_center of box1
        float w1 = boxesGm.GetValue(box1_offset + 2);   // width of box1
        float h1 = boxesGm.GetValue(box1_offset + 3);   // height of box1

        float cx2 = boxesGm.GetValue(box2_offset);      // x_center of box2
        float cy2 = boxesGm.GetValue(box2_offset + 1);  // y_center of box2
        float w2 = boxesGm.GetValue(box2_offset + 2);   // width of box2
        float h2 = boxesGm.GetValue(box2_offset + 3);   // height of box2

        // 将中心模式转换为角点模式
        float x1_1 = cx1 - w1 / 2;
        float y1_1 = cy1 - h1 / 2;
        float x2_1 = cx1 + w1 / 2;
        float y2_1 = cy1 + h1 / 2;

        float x1_2 = cx2 - w2 / 2;
        float y1_2 = cy2 - h2 / 2;
        float x2_2 = cx2 + w2 / 2;
        float y2_2 = cy2 + h2 / 2;

        // 确保 x1 和 x2 是对角线的两个角点
        float box1_xmin = min(x1_1, x2_1);
        float box1_xmax = max(x1_1, x2_1);
        float box1_ymin = min(y1_1, y2_1);
        float box1_ymax = max(y1_1, y2_1);

        float box2_xmin = min(x1_2, x2_2);
        float box2_xmax = max(x1_2, x2_2);
        float box2_ymin = min(y1_2, y2_2);
        float box2_ymax = max(y1_2, y2_2);

        // 计算 IoU
        float x1 = max(box1_xmin, box2_xmin);  // 交集的左边界
        float y1 = max(box1_ymin, box2_ymin);  // 交集的上边界
        float x2 = min(box1_xmax, box2_xmax);  // 交集的右边界
        float y2 = min(box1_ymax, box2_ymax);  // 交集的下边界

        float w = max(static_cast<float>(0), x2 - x1);  // 交集的宽度
        float h = max(static_cast<float>(0), y2 - y1);  // 交集的高度
        float inter = w * h;  // 交集面积

        float area1 = (box1_xmax - box1_xmin) * (box1_ymax - box1_ymin);  // box1 的面积
        float area2 = (box2_xmax - box2_xmin) * (box2_ymax - box2_ymin);  // box2 的面积

        // 计算并集面积
        float union_area = area1 + area2 - inter;

        // 计算 IoU
        return inter / union_area;
    }

   


private:
    AscendC::GlobalTensor<float> boxesGm;
    AscendC::GlobalTensor<float> scoresGm;
    AscendC::GlobalTensor<int32_t> max_output_boxes_per_classGm;
    AscendC::GlobalTensor<float> iou_thresholdGm;
    AscendC::GlobalTensor<float> score_thresholdGm;
    AscendC::GlobalTensor<int32_t> selected_indicesGm;


    int32_t max_output_boxes_per_class; // int32_t
    float iou_threshold; // float
    float score_threshold; // float

    uint32_t num_batches;
    uint32_t spatial_dimension;
    uint32_t num_classes;
    uint32_t num_selected_indices;
    uint32_t center_point_box;

};

extern "C" __global__ __aicore__ void non_max_suppression(GM_ADDR boxes, GM_ADDR scores, GM_ADDR max_output_boxes_per_class, 
    GM_ADDR iou_threshold, GM_ADDR score_threshold, 
    GM_ADDR selected_indices, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    NonMaxSuppressionKernel<DTYPE_BOXES> op;
    op.Init(boxes, scores, max_output_boxes_per_class, 
        iou_threshold, score_threshold, selected_indices,
        tiling_data.num_batches, tiling_data.spatial_dimension, 
        tiling_data.num_classes, tiling_data.num_selected_indices,
        tiling_data.center_point_box
    );
    op.Process();
}