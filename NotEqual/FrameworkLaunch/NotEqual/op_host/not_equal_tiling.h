/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef NotEqual_TILING_H // 防止重复包含的包含保护宏
#define NotEqual_TILING_H
#include "register/tilingdata_base.h" // 引入基础平铺数据头文件

namespace optiling { // 定义平铺操作的命名空间
BEGIN_TILING_DATA_DEF(TilingData) // 开始定义 TilingData 结构
  TILING_DATA_FIELD_DEF(uint32_t, totalLength); // 平铺总长度字段
  TILING_DATA_FIELD_DEF(uint32_t, tileNumMean); // 平铺数量均值字段
  TILING_DATA_FIELD_DEF(uint32_t, tileNumEnd); // 平铺数量结束字段
  TILING_DATA_FIELD_DEF(uint32_t, tileLengthMean); // 平铺长度均值字段
  TILING_DATA_FIELD_DEF(uint32_t, tileLengthEnd); // 平铺长度结束字段
  TILING_DATA_FIELD_DEF(uint32_t, blockLengthMean); // 块长度均值字段
  TILING_DATA_FIELD_DEF(uint32_t, blockLengthEnd); // 块长度结束字段

  TILING_DATA_FIELD_DEF(uint32_t, x1_length);
  TILING_DATA_FIELD_DEF(uint32_t, x2_length);
  TILING_DATA_FIELD_DEF(int64_t, DimNum);
  TILING_DATA_FIELD_DEF_ARR(int64_t, 128, shape);  
  TILING_DATA_FIELD_DEF_ARR(int64_t, 64, shapefull);
END_TILING_DATA_DEF; // 结束 TilingData 结构的定义

REGISTER_TILING_DATA_CLASS(NotEqual, TilingData) // 注册 NotEqual 类与 TilingData 结构
} // 结束命名空间 optiling
#endif // NotEqual_TILING_H