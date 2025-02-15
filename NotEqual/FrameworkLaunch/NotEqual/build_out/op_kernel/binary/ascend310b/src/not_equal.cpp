/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 *
 * Function : z = x + y
 * This sample is a very basic sample that implements vector add on Ascend
 * plaform.
 */
#include <type_traits>
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;  // tensor num for each queue
constexpr float NEGATIVE_ONE_FP32 = -1.0F;
constexpr float POSITIVE_ONE_FP32 = 1.0F;
constexpr int32_t NEGATIVE_ONE_I32 = -1;
constexpr int32_t POSITIVE_ONE_I32 = 1;
constexpr float MIN_ACCURACY_FP16 = 0.00000005960464477539063F;
constexpr float MAX_MUL_FP16 = 4096;
// constexpr float MIN_ACCURACY_FP16 = 0.00006103515625F;  // 正规数范围的最小值
// constexpr float MAX_MUL_FP16 = 128;
constexpr float MIN_ACCURACY_FP32 = 1.1754943508222875e-38;
constexpr float MAX_MUL_1_FP32 = 1125899906842624;
constexpr float MAX_MUL_2_FP32 = 67108864;
constexpr uint32_t BLOCK_SIZE = 32;

constexpr int32_t inputVarNum = 2; // 输入个数
constexpr int32_t maxDimNum = 64; // 最大维度数量

template <typename typeT>
class KernelNotEqual {
public:
  __aicore__ inline KernelNotEqual() {}
  __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
    uint32_t total_length, uint32_t tile_num_mean,
    uint32_t tile_num_end, uint32_t tile_length_mean,
    uint32_t tile_length_end, uint32_t block_length_mean,
    uint32_t block_length_end) {
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    ResovleTiling(total_length, tile_num_mean, tile_num_end, tile_length_mean,
      tile_length_end, block_length_mean, block_length_end);
    x1_gm.SetGlobalBuffer(
      (__gm__ typeT*)x1 + this->block_offset * AscendC::GetBlockIdx(),
      this->block_length);
    x2_gm.SetGlobalBuffer(
      (__gm__ typeT*)x2 + this->block_offset * AscendC::GetBlockIdx(),
      this->block_length);
    y_gm.SetGlobalBuffer((__gm__ int8_t*)y + this->block_offset * AscendC::GetBlockIdx(),
      this->block_length);


    pipe.InitBuffer(x1_inque, BUFFER_NUM, this->tile_cache * sizeof(typeT));
    pipe.InitBuffer(x2_inque, BUFFER_NUM, this->tile_cache * sizeof(typeT));
    pipe.InitBuffer(y_outque, BUFFER_NUM,
      this->tile_cache * sizeof(int8_t) < BLOCK_SIZE
      ? BLOCK_SIZE
      : this->tile_cache * sizeof(int8_t));
    pipe.InitBuffer(calc_buf_1, this->tile_cache * sizeof(typeT));
    pipe.InitBuffer(calc_buf_2, this->tile_cache * sizeof(half) < BLOCK_SIZE
      ? BLOCK_SIZE
      : this->tile_cache * sizeof(half));
    pipe.InitBuffer(calc_buf_3, this->tile_cache * sizeof(half) < BLOCK_SIZE
      ? BLOCK_SIZE
      : this->tile_cache * sizeof(half));
    pipe.InitBuffer(calc_buf_4, this->tile_cache * sizeof(float) < BLOCK_SIZE
      ? BLOCK_SIZE 
      : this->tile_cache * sizeof(float));

  }
  __aicore__ inline void Process() {
    if (this->total_length <= BLOCK_SIZE / sizeof(typeT)) {
      CopyInPad(0);
      Compute(0);
      CopyOutPad(0);
      return;
    }
    int32_t loopCount = this->tile_num;
    for (int32_t i = 0; i < loopCount - 1; i++) {
      CopyIn(i);
      Compute(i);
      CopyOut(i);
    }
    if (AscendC::GetBlockIdx() == (AscendC::GetBlockNum() - 1)) {
      CopyInPad(loopCount - 1);
      Compute(loopCount - 1);
      CopyOutPad(loopCount - 1);
    }
    else {
      CopyIn(loopCount - 1);
      Compute(loopCount - 1);
      CopyOut(loopCount - 1);
    }
  }

private:
  __aicore__ inline void ResovleTiling(
    uint32_t total_length, uint32_t tile_num_mean, uint32_t tile_num_end,
    uint32_t tile_length_mean, uint32_t tile_length_end, uint32_t block_length_mean,
    uint32_t block_length_end) {
    uint32_t pad32 = BLOCK_SIZE;  // 对齐32B需要的最小数据量
    this->total_length = total_length;
    if (AscendC::GetBlockNum() >= 1 && AscendC::GetBlockIdx() == (AscendC::GetBlockNum() - 1)) {
      this->block_length = block_length_end;
      this->tile_num = tile_num_end;
    }
    else {
      this->block_length = block_length_mean;
      this->tile_num = tile_num_mean;
    }
    this->block_offset = block_length_mean;
    this->tile_length = tile_length_mean;
    this->tile_cache = tile_length_mean;
    this->tile_length_end = tile_length_end;
    if (total_length < pad32) {
      this->block_offset = 0;
      this->tile_cache = pad32;
      this->block_length = pad32;
    }
  }
  __aicore__ inline void CopyIn(int32_t progress) {
    AscendC::LocalTensor<typeT> x1_local = x1_inque.AllocTensor<typeT>();
    AscendC::LocalTensor<typeT> x2_local = x2_inque.AllocTensor<typeT>();
    AscendC::DataCopy(x1_local, x1_gm[progress * this->tile_cache], this->tile_cache);
    AscendC::DataCopy(x2_local, x2_gm[progress * this->tile_cache], this->tile_cache);
    x1_inque.EnQue(x1_local);
    x2_inque.EnQue(x2_local);
  }
  __aicore__ inline void CopyInPad(int32_t progress) {
    AscendC::LocalTensor<typeT> x1_local = x1_inque.AllocTensor<typeT>();
    AscendC::LocalTensor<typeT> x2_local = x2_inque.AllocTensor<typeT>();
    AscendC::DataCopy(x1_local, x1_gm[progress * this->tile_cache],
      ((this->tile_length_end + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE));
    AscendC::DataCopy(x2_local, x2_gm[progress * this->tile_cache],
      ((this->tile_length_end + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE));
    x1_inque.EnQue(x1_local);
    x2_inque.EnQue(x2_local);
  }
  __aicore__ inline void Compute(int32_t progress) {
    AscendC::LocalTensor<typeT> x1_local = x1_inque.DeQue<typeT>();
    AscendC::LocalTensor<typeT> x2_local = x2_inque.DeQue<typeT>();
    AscendC::LocalTensor<int8_t> y_local = y_outque.AllocTensor<int8_t>();
    AscendC::LocalTensor<typeT> y_compute = calc_buf_1.Get<typeT>();
    // AscendC::DumpTensor(x1_local, 0,this->tile_cache); 
    // AscendC::DumpTensor(x2_local, 0, this->tile_cache); 
    if constexpr (std::is_same_v<typeT, half>) {
      // try 1: 
      AscendC::Sub(y_compute, x1_local, x2_local, this->tile_cache);
      AscendC::Abs(y_compute, y_compute, this->tile_cache);
      AscendC::Mins(y_compute, y_compute, (half)MIN_ACCURACY_FP16, this->tile_cache);
      AscendC::Muls(y_compute, y_compute, (half)MAX_MUL_FP16, this->tile_cache);
      AscendC::Muls(y_compute, y_compute, (half)MAX_MUL_FP16, this->tile_cache);
      AscendC::Cast(y_local, y_compute, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      // try 2:
      // AscendC::Compare(y_local, x1_local, x2_local, AscendC::CMPMODE::NE, this->tile_cache);
    }
    else if constexpr (std::is_same_v<typeT, float>) {
      // try 1:
      AscendC::LocalTensor<half> y_fp16 = calc_buf_2.Get<half>();
      AscendC::Sub(y_compute, x1_local, x2_local, this->tile_cache);
      AscendC::Abs(y_compute, y_compute, this->tile_cache);
      AscendC::Mins(y_compute, y_compute, (float)MIN_ACCURACY_FP32, this->tile_cache);
      AscendC::Muls(y_compute, y_compute, (float)MAX_MUL_1_FP32, this->tile_cache);
      AscendC::Muls(y_compute, y_compute, (float)MAX_MUL_1_FP32, this->tile_cache);
      AscendC::Muls(y_compute, y_compute, (float)MAX_MUL_2_FP32, this->tile_cache);
      AscendC::Cast(y_fp16, y_compute, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      AscendC::Cast(y_local, y_fp16, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      // try 2:
      // AscendC::Compare(y_local, x1_local, x2_local, AscendC::CMPMODE::NE, this->tile_cache);

    }
    else if constexpr (std::is_same_v<typeT, int8_t>) {
      // try 1:
      AscendC::LocalTensor<half> x1_local_fp16 = calc_buf_2.Get<half>();
      AscendC::LocalTensor<half> x2_local_fp16 = calc_buf_3.Get<half>();
      AscendC::LocalTensor<half> y_local_fp16 = calc_buf_4.Get<half>();
      AscendC::Cast(x1_local_fp16, x1_local, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      AscendC::Cast(x2_local_fp16, x2_local, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      AscendC::Sub(y_local_fp16, x1_local_fp16, x2_local_fp16, this->tile_cache);
      AscendC::Abs(y_local_fp16, y_local_fp16, this->tile_cache);
      AscendC::Mins(y_local_fp16, y_local_fp16, (half)MIN_ACCURACY_FP16, this->tile_cache);
      AscendC::Muls(y_local_fp16, y_local_fp16, (half)MAX_MUL_FP16, this->tile_cache);
      AscendC::Muls(y_local_fp16, y_local_fp16, (half)MAX_MUL_FP16, this->tile_cache);
      AscendC::Cast(y_local, y_local_fp16, AscendC::RoundMode::CAST_NONE, this->tile_cache);

      // try 2:
      // AscendC::Cast(x1_local_fp16, x1_local, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      // AscendC::Cast(x2_local_fp16, x2_local, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      // AscendC::Compare(y_local, x1_local_fp16, x2_local_fp16, AscendC::CMPMODE::NE, this->tile_cache);

      // try 3: 不能简单的这样操作，会出问题。
      // AscendC::Sub(y_local, x1_local, x2_local, this->tile_cache);
    }
    else if constexpr (std::is_same_v<typeT, int32_t>) {
      AscendC::LocalTensor<half> y_fp16 = calc_buf_3.Get<half>();
      AscendC::LocalTensor<float> y_fp32 = calc_buf_4.Get<float>();
      // try 1: 
      // AscendC::Sub(x1_local, x1_local, x2_local, this->tile_cache); 
      // AscendC::Mins(x1_local, x1_local, (int32_t)POSITIVE_ONE_I32, this->tile_cache);
      // AscendC::Maxs(x1_local, x1_local, (int32_t)NEGATIVE_ONE_I32, this->tile_cache);
      // AscendC::Mul(y_compute, x1_local, x1_local, this->tile_cache);
      // AscendC::Cast(y_fp32, y_compute, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      // AscendC::Cast(y_fp16, y_fp32, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      // AscendC::Cast(y_local, y_fp16, AscendC::RoundMode::CAST_NONE, this->tile_cache);

      // try 2:
      // 计算 x1 - x2 的绝对值
      AscendC::Sub(y_compute, x1_local, x2_local, this->tile_cache);
      AscendC::Abs(y_compute, y_compute, this->tile_cache);
      // 将绝对值限制为 1
      AscendC::Mins(y_compute, y_compute, (int32_t)POSITIVE_ONE_I32, this->tile_cache);
      // 将结果写入 y_fp32 和 y_fp16
      AscendC::Cast(y_fp32, y_compute, AscendC::RoundMode::CAST_CEIL, this->tile_cache);
      AscendC::Cast(y_fp16, y_fp32, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      AscendC::Cast(y_local, y_fp16, AscendC::RoundMode::CAST_NONE, this->tile_cache);

      // try 3: 
      // AscendC::LocalTensor<int16_t> x1_local_int16 = calc_buf_5.Get<int16_t>();
      // int32_t offset1 = this->tile_cache;
      // int32_t offset2 = this->tile_cache * 2;
      // AscendC::LocalTensor<int16_t> x2_local_int16 = x1_local_int16[offset1];
      // AscendC::LocalTensor<int16_t> y_local_int16 = x1_local_int16[offset2];
      // // 转为 int16
      // AscendC::Cast(x1_local_int16, x1_local, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      // AscendC::Cast(x2_local_int16, x2_local, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      // // A and B 
      // AscendC::And(y_local_int16, x1_local_int16, x2_local_int16, this->tile_cache);
      // // not(A and B)
      // AscendC::Not(y_local_int16, y_local_int16, this->tile_cache);
      // // A or B 
      // AscendC::Or(x1_local_int16, x1_local_int16, x2_local_int16, this->tile_cache);
      // // not(A and B) and (A or B)
      // AscendC::And(y_local_int16, y_local_int16, x1_local_int16, this->tile_cache);
      // // int16 to fp16(只能这样转) , fp16 -> int8
      // AscendC::Cast(y_fp16, y_local_int16, AscendC::RoundMode::CAST_ROUND, this->tile_cache);
      // AscendC::Cast(y_local, y_fp16, AscendC::RoundMode::CAST_NONE, this->tile_cache);
    }

    y_outque.EnQue<int8_t>(y_local);
    x1_inque.FreeTensor(x1_local);
    x2_inque.FreeTensor(x2_local);
  }
  __aicore__ inline void CopyOut(int32_t progress) {
    AscendC::LocalTensor<int8_t> y_local = y_outque.DeQue<int8_t>();
    AscendC::DataCopy(y_gm[progress * this->tile_cache], y_local,
      this->tile_cache);
    y_outque.FreeTensor(y_local);
  }
  __aicore__ inline void CopyOutPad(int32_t progress) {
    AscendC::LocalTensor<int8_t> y_local = y_outque.DeQue<int8_t>();
    AscendC::DataCopy(y_gm[progress * this->tile_cache], y_local,
      (this->tile_length_end + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE);
    y_outque.FreeTensor(y_local);
  }

private:
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf_1, calc_buf_2, calc_buf_3, calc_buf_4; 
  AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf_5; 
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> x1_inque, x2_inque;
  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> y_outque;
  AscendC::GlobalTensor<typeT> x1_gm, x2_gm;
  AscendC::GlobalTensor<int8_t> y_gm;
  uint32_t total_length, block_length, block_offset, tile_num, tile_cache,
    tile_length, tile_length_end;
};

template <typename typeT>
class KernelNotEqual_Broadcast {
public:
  __aicore__ inline KernelNotEqual_Broadcast() {}

  __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
    uint32_t total_length, uint32_t tile_num_mean,
    uint32_t tile_num_end, uint32_t tile_length_mean,
    uint32_t tile_length_end, uint32_t block_length_mean,
    uint32_t block_length_end, uint32_t x1_length, uint32_t x2_length,int64_t DimNum, int64_t ss[], int64_t sf[]) {
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

    this->x1_length = x1_length;
    this->x2_length = x2_length;
    ResovleTiling(total_length, tile_num_mean, tile_num_end, tile_length_mean,
      tile_length_end, block_length_mean, block_length_end);
    
    this->DimNum = DimNum; // 维度数量
    for (int i = 0; i < DimNum; ++i) {
        ((int64_t *)this->shape)[i] = ss[i];
        ((int64_t *)this->shape)[maxDimNum + i] = ss[maxDimNum + i];
        ((int64_t *)this->shapefull)[i] = sf[i];
    }


    // Print_Param();
    // 将其中数量最少的数据全部放进来，需要额外读取。
    if(this->x1_length < this->total_length){
      x1_gm.SetGlobalBuffer( (__gm__ typeT*)x1, this->total_length); // 加载全部数据 (需要广播)
    }else{
      x1_gm.SetGlobalBuffer(
        (__gm__ typeT*)x1 + this->block_offset * AscendC::GetBlockIdx(),  // 只需要加载 block 内数据(由于没有广播，block内数据是相同的)
        this->block_length);
    }
    if(this->x2_length < this->total_length){
      x2_gm.SetGlobalBuffer( (__gm__ typeT*)x2, this->total_length);
    }else{
        x2_gm.SetGlobalBuffer(
          (__gm__ typeT*)x2 + this->block_offset * AscendC::GetBlockIdx(),
          this->block_length);
    }
    y_gm.SetGlobalBuffer((__gm__ int8_t*)y + this->block_offset * AscendC::GetBlockIdx(),
      this->block_length);


    pipe.InitBuffer(x1_inque, BUFFER_NUM, this->tile_cache * sizeof(typeT));
    pipe.InitBuffer(x2_inque, BUFFER_NUM, this->tile_cache * sizeof(typeT));
    pipe.InitBuffer(y_outque, BUFFER_NUM,
      this->tile_cache * sizeof(int8_t) < BLOCK_SIZE
      ? BLOCK_SIZE
      : this->tile_cache * sizeof(int8_t));
    pipe.InitBuffer(calc_buf_1, this->tile_cache * sizeof(typeT));
    pipe.InitBuffer(calc_buf_2, this->tile_cache * sizeof(half) < BLOCK_SIZE
      ? BLOCK_SIZE
      : this->tile_cache * sizeof(half));
    pipe.InitBuffer(calc_buf_3, this->tile_cache * sizeof(half) < BLOCK_SIZE
      ? BLOCK_SIZE
      : this->tile_cache * sizeof(half));
    pipe.InitBuffer(calc_buf_4, this->tile_cache * sizeof(float) < BLOCK_SIZE
      ? BLOCK_SIZE 
      : this->tile_cache * sizeof(float));
  }

  __aicore__ inline void Process() {
    if (this->total_length <= BLOCK_SIZE / sizeof(typeT)) {
      CopyInPad(0);
      Compute(0);
      CopyOutPad(0);
      return;
    }
    int32_t loopCount = this->tile_num;
    for (int32_t i = 0; i < loopCount - 1; i++) {
      CopyIn(i);
      Compute(i);
      CopyOut(i);
    }
    if (AscendC::GetBlockIdx() == (AscendC::GetBlockNum() - 1)) {
      CopyInPad(loopCount - 1);
      Compute(loopCount - 1);
      CopyOutPad(loopCount - 1);
    }
    else {
      CopyIn(loopCount - 1);
      Compute(loopCount - 1);
      CopyOut(loopCount - 1);
    }
  }


private:
  __aicore__ inline void ResovleTiling(
    uint32_t total_length, uint32_t tile_num_mean, uint32_t tile_num_end,
    uint32_t tile_length_mean, uint32_t tile_length_end, uint32_t block_length_mean,
    uint32_t block_length_end) {
    uint32_t pad32 = BLOCK_SIZE;  // 对齐32B需要的最小数据量
    this->total_length = total_length;
    // 如果是最后一个核，就直接让 block_length = block_length_end
    if (AscendC::GetBlockNum() >= 1 && AscendC::GetBlockIdx() == (AscendC::GetBlockNum() - 1)) {
      this->block_length = block_length_end;
      this->tile_num = tile_num_end;
    }
    else {
      this->block_length = block_length_mean;
      this->tile_num = tile_num_mean;
    }
    this->block_offset = block_length_mean;
    this->tile_length = tile_length_mean;
    this->tile_cache = tile_length_mean;
    this->tile_length_end = tile_length_end;
    if (total_length < pad32) {
      this->block_offset = 0;
      this->tile_cache = pad32;
      this->block_length = pad32;
    }
  }
  __aicore__ inline void CopyIn(int32_t progress) {
    // progress 表示 block 中的第几个 tile
    AscendC::LocalTensor<typeT> x1_local = x1_inque.AllocTensor<typeT>();
    AscendC::LocalTensor<typeT> x2_local = x2_inque.AllocTensor<typeT>();
    // 对 x1 进行广播
    if(this->x1_length < this->total_length){// offset 是 block的 offset + till的 offset
      BroadCX1(x1_local, this->block_offset * AscendC::GetBlockIdx() + progress * this->tile_cache, this->tile_cache);
    }else{
      AscendC::DataCopy(x1_local, x1_gm[progress * this->tile_cache], this->tile_cache); 
    }
    if(this->x2_length < this->total_length){
      BroadCX2(x2_local, this->block_offset * AscendC::GetBlockIdx() + progress * this->tile_cache, this->tile_cache);
    }else{
      AscendC::DataCopy(x2_local, x2_gm[progress * this->tile_cache], this->tile_cache);
    }
    x1_inque.EnQue(x1_local);
    x2_inque.EnQue(x2_local);
  }

  __aicore__ inline void CopyInPad(int32_t progress) {
    AscendC::LocalTensor<typeT> x1_local = x1_inque.AllocTensor<typeT>();
    AscendC::LocalTensor<typeT> x2_local = x2_inque.AllocTensor<typeT>();

    if(this->x1_length < this->total_length){// offset 是 block的 offset + till的 offset
      BroadCX1(x1_local, this->block_offset * AscendC::GetBlockIdx() + progress * this->tile_cache, ((this->tile_length_end + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE));
    }else{
      AscendC::DataCopy(x1_local, x1_gm[progress * this->tile_cache],
        ((this->tile_length_end + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE));
    }

    if(this->x2_length < this->total_length){
      BroadCX2(x2_local, this->block_offset * AscendC::GetBlockIdx() + progress * this->tile_cache, ((this->tile_length_end + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE));  
    }else{
      AscendC::DataCopy(x2_local, x2_gm[progress * this->tile_cache],
        ((this->tile_length_end + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE));
    }
    x1_inque.EnQue(x1_local);
    x2_inque.EnQue(x2_local);
  }
  __aicore__ inline void Compute(int32_t progress) {
    AscendC::LocalTensor<typeT> x1_local = x1_inque.DeQue<typeT>();
    AscendC::LocalTensor<typeT> x2_local = x2_inque.DeQue<typeT>();
    AscendC::LocalTensor<int8_t> y_local = y_outque.AllocTensor<int8_t>();
    AscendC::LocalTensor<typeT> y_compute = calc_buf_1.Get<typeT>();
    // AscendC::DumpTensor(x1_local, 0,this->tile_cache); 
    // AscendC::DumpTensor(x2_local, 0, this->tile_cache); 
    if constexpr (std::is_same_v<typeT, half>) {
      // try 1: 
      AscendC::Sub(y_compute, x1_local, x2_local, this->tile_cache);
      AscendC::Abs(y_compute, y_compute, this->tile_cache);
      AscendC::Mins(y_compute, y_compute, (half)MIN_ACCURACY_FP16, this->tile_cache);
      AscendC::Muls(y_compute, y_compute, (half)MAX_MUL_FP16, this->tile_cache);
      AscendC::Muls(y_compute, y_compute, (half)MAX_MUL_FP16, this->tile_cache);
      AscendC::Cast(y_local, y_compute, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      // try 2:
      // AscendC::Compare(y_local, x1_local, x2_local, AscendC::CMPMODE::NE, this->tile_cache);
    }
    else if constexpr (std::is_same_v<typeT, float>) {
      // try 1:
      AscendC::LocalTensor<half> y_fp16 = calc_buf_2.Get<half>();
      AscendC::Sub(y_compute, x1_local, x2_local, this->tile_cache);
      AscendC::Abs(y_compute, y_compute, this->tile_cache);
      AscendC::Mins(y_compute, y_compute, (float)MIN_ACCURACY_FP32, this->tile_cache);
      AscendC::Muls(y_compute, y_compute, (float)MAX_MUL_1_FP32, this->tile_cache);
      AscendC::Muls(y_compute, y_compute, (float)MAX_MUL_1_FP32, this->tile_cache);
      AscendC::Muls(y_compute, y_compute, (float)MAX_MUL_2_FP32, this->tile_cache);
      AscendC::Cast(y_fp16, y_compute, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      AscendC::Cast(y_local, y_fp16, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      // try 2:
      // AscendC::Compare(y_local, x1_local, x2_local, AscendC::CMPMODE::NE, this->tile_cache);

    }
    else if constexpr (std::is_same_v<typeT, int8_t>) {
      // try 1:
      AscendC::LocalTensor<half> x1_local_fp16 = calc_buf_2.Get<half>();
      AscendC::LocalTensor<half> x2_local_fp16 = calc_buf_3.Get<half>();
      AscendC::LocalTensor<half> y_local_fp16 = calc_buf_4.Get<half>();
      AscendC::Cast(x1_local_fp16, x1_local, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      AscendC::Cast(x2_local_fp16, x2_local, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      AscendC::Sub(y_local_fp16, x1_local_fp16, x2_local_fp16, this->tile_cache);
      AscendC::Abs(y_local_fp16, y_local_fp16, this->tile_cache);
      AscendC::Mins(y_local_fp16, y_local_fp16, (half)MIN_ACCURACY_FP16, this->tile_cache);
      AscendC::Muls(y_local_fp16, y_local_fp16, (half)MAX_MUL_FP16, this->tile_cache);
      AscendC::Muls(y_local_fp16, y_local_fp16, (half)MAX_MUL_FP16, this->tile_cache);
      AscendC::Cast(y_local, y_local_fp16, AscendC::RoundMode::CAST_NONE, this->tile_cache);

      // try 2:
      // AscendC::Cast(x1_local_fp16, x1_local, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      // AscendC::Cast(x2_local_fp16, x2_local, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      // AscendC::Compare(y_local, x1_local_fp16, x2_local_fp16, AscendC::CMPMODE::NE, this->tile_cache);

      // try 3: 不能简单的这样操作，会出问题。
      // AscendC::Sub(y_local, x1_local, x2_local, this->tile_cache);
    }
    else if constexpr (std::is_same_v<typeT, int32_t>) {
      AscendC::LocalTensor<half> y_fp16 = calc_buf_3.Get<half>();
      AscendC::LocalTensor<float> y_fp32 = calc_buf_4.Get<float>();
      // try 1: 
      // AscendC::Sub(x1_local, x1_local, x2_local, this->tile_cache); 
      // AscendC::Mins(x1_local, x1_local, (int32_t)POSITIVE_ONE_I32, this->tile_cache);
      // AscendC::Maxs(x1_local, x1_local, (int32_t)NEGATIVE_ONE_I32, this->tile_cache);
      // AscendC::Mul(y_compute, x1_local, x1_local, this->tile_cache);
      // AscendC::Cast(y_fp32, y_compute, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      // AscendC::Cast(y_fp16, y_fp32, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      // AscendC::Cast(y_local, y_fp16, AscendC::RoundMode::CAST_NONE, this->tile_cache);

      // try 2:
      // 计算 x1 - x2 的绝对值
      AscendC::Sub(y_compute, x1_local, x2_local, this->tile_cache);
      AscendC::Abs(y_compute, y_compute, this->tile_cache);
      // 将绝对值限制为 1
      AscendC::Mins(y_compute, y_compute, (int32_t)POSITIVE_ONE_I32, this->tile_cache);
      // 将结果写入 y_fp32 和 y_fp16
      AscendC::Cast(y_fp32, y_compute, AscendC::RoundMode::CAST_CEIL, this->tile_cache);
      AscendC::Cast(y_fp16, y_fp32, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      AscendC::Cast(y_local, y_fp16, AscendC::RoundMode::CAST_NONE, this->tile_cache);

      // try 3: 
      // AscendC::LocalTensor<int16_t> x1_local_int16 = calc_buf_5.Get<int16_t>();
      // int32_t offset1 = this->tile_cache;
      // int32_t offset2 = this->tile_cache * 2;
      // AscendC::LocalTensor<int16_t> x2_local_int16 = x1_local_int16[offset1];
      // AscendC::LocalTensor<int16_t> y_local_int16 = x1_local_int16[offset2];
      // // 转为 int16
      // AscendC::Cast(x1_local_int16, x1_local, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      // AscendC::Cast(x2_local_int16, x2_local, AscendC::RoundMode::CAST_NONE, this->tile_cache);
      // // A and B 
      // AscendC::And(y_local_int16, x1_local_int16, x2_local_int16, this->tile_cache);
      // // not(A and B)
      // AscendC::Not(y_local_int16, y_local_int16, this->tile_cache);
      // // A or B 
      // AscendC::Or(x1_local_int16, x1_local_int16, x2_local_int16, this->tile_cache);
      // // not(A and B) and (A or B)
      // AscendC::And(y_local_int16, y_local_int16, x1_local_int16, this->tile_cache);
      // // int16 to fp16(只能这样转) , fp16 -> int8
      // AscendC::Cast(y_fp16, y_local_int16, AscendC::RoundMode::CAST_ROUND, this->tile_cache);
      // AscendC::Cast(y_local, y_fp16, AscendC::RoundMode::CAST_NONE, this->tile_cache);
    }

    y_outque.EnQue<int8_t>(y_local);
    x1_inque.FreeTensor(x1_local);
    x2_inque.FreeTensor(x2_local);
  }
  __aicore__ inline void CopyOut(int32_t progress) {
    AscendC::LocalTensor<int8_t> y_local = y_outque.DeQue<int8_t>();
    AscendC::DataCopy(y_gm[progress * this->tile_cache], y_local,
      this->tile_cache);
    y_outque.FreeTensor(y_local);
  }
  __aicore__ inline void CopyOutPad(int32_t progress) {
    AscendC::LocalTensor<int8_t> y_local = y_outque.DeQue<int8_t>();
    AscendC::DataCopy(y_gm[progress * this->tile_cache], y_local,
      (this->tile_length_end + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE);
    y_outque.FreeTensor(y_local);
  }
  __aicore__ inline void BroadCX1(AscendC::LocalTensor<typeT> &dst, uint32_t offset, uint32_t length) {
    // dst: local tensor, 长度是 length， 需要填充的数据，将根据 pre_index 对应的数据进行填充
    // offset : 是 广播后tensor 的 flat_index; 
    // length : 需要找到的数据量
    // return : 需要找到 flat_index 对应广播前的 pre_index;
      if(this->x1_length == 1) {
          typeT tmp = x1_gm.GetValue(0);
          if constexpr (std::is_same_v<typeT, int8_t>){
            AscendC::LocalTensor<half> tmp_fp16 = calc_buf_2.Get<half>();
            AscendC::Duplicate(tmp_fp16, (half)tmp, length);
            AscendC::Cast(dst, tmp_fp16, AscendC::RoundMode::CAST_NONE, length);
          }
          else{
            AscendC::Duplicate(dst, tmp, length);
          }
          return;
      }
      for(uint32_t i = 0; i < length; i++) {
          int istart = i + offset; // 广播后的 flat_index
          int idxtmp = GetOriginalIndex(istart, 0); // 广播前的 pre_index
          typeT tmp = x1_gm.GetValue(idxtmp); // 根据 pre_index 获取值
          dst.SetValue(i, tmp); // 设置值
      }
  }
  __aicore__ inline void BroadCX2(AscendC::LocalTensor<typeT> &dst, uint32_t offset, uint32_t length) {
      if(this->x2_length == 1) {
          typeT tmp = x2_gm.GetValue(0);
          if constexpr (std::is_same_v<typeT, int8_t>){
            AscendC::LocalTensor<half> tmp_fp16 = calc_buf_2.Get<half>();
            AscendC::Duplicate(tmp_fp16, (half)tmp, length);
            AscendC::Cast(dst, tmp_fp16, AscendC::RoundMode::CAST_NONE, length);
          }
          else{
            AscendC::Duplicate(dst, tmp, length);
          }          
          return;
      }
      for(uint32_t i = 0; i < length; i++) {
          int istart = i + offset;
          int idxtmp = GetOriginalIndex(istart, 1);
          typeT tmp = x2_gm.GetValue(idxtmp);
          dst.SetValue(i, tmp);
      }
  }

  __aicore__ inline int GetOriginalIndex(int idx_broad, int inputNo) {
      int idx_org = 0;
      for (int k = 1; k <= this->DimNum; k++) {
          int kpos = 0;
          int krange = 1;
          if (k < this->DimNum) {
              for (int m = k + 1; m <= this->DimNum; m++) {
                  krange *= this->shapefull[m - 1];
              }
              kpos = idx_broad / krange;
              idx_broad = idx_broad % krange;
          } else {
              krange = this->shapefull[k - 1];
              kpos = idx_broad % krange;
          }

          int krangeB = 1;
          if (this->shapefull[k - 1] == this->shape[inputNo][k - 1]) {
              if (k < this->DimNum) {
                  for (int m = k + 1; m <= this->DimNum; m++) {
                      krangeB *= this->shape[inputNo][m - 1];
                  }
                  idx_org += kpos * krangeB;
              } else {
                  idx_org += kpos;
              }
          }
      }
      return idx_org;
  }
  __aicore__ inline void Print_Param(){
    for (int i = 0; i < DimNum; ++i) {
      AscendC::PRINTF("shape[%d]: %ld", i, this->shape[0][i]);
      AscendC::PRINTF("shape[%d + maxDimNum]: %ld", i, this->shape[1][i]);
      AscendC::PRINTF("shapefull[%d]: %ld", i, this->shapefull[i]);
    }
    AscendC::PRINTF("DimNum:%d", this->DimNum);
    AscendC::PRINTF("total_length:%d", this->total_length);
    AscendC::PRINTF("block_length:%d", this->block_length);
    AscendC::PRINTF("block_offset:%d", this->block_offset);
    AscendC::PRINTF("tile_num:%d", this->tile_num);
    AscendC::PRINTF("tile_cache:%d", this->tile_cache);
    AscendC::PRINTF("tile_length:%d", this->tile_length);
    AscendC::PRINTF("tile_length_end:%d", this->tile_length_end);
    AscendC::PRINTF("x1_length:%d", this->x1_length);
    AscendC::PRINTF("x2_length:%d", this->x2_length);
  }
private:
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf_1, calc_buf_2, calc_buf_3, calc_buf_4; 
  AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf_5; 
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> x1_inque, x2_inque;
  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> y_outque;
  AscendC::GlobalTensor<typeT> x1_gm, x2_gm;
  AscendC::GlobalTensor<int8_t> y_gm;
  uint32_t total_length, block_length, block_offset, tile_num, tile_cache,
    tile_length, tile_length_end;
  uint32_t x1_length, x2_length;
  int64_t DimNum;
  int64_t shape[2][64];
  int64_t shapefull[maxDimNum];
};

extern "C" __global__ __aicore__ void not_equal(GM_ADDR x1, GM_ADDR x2,
  GM_ADDR y, GM_ADDR workspace,GM_ADDR tiling) {

  GET_TILING_DATA(tiling_data, tiling);
  if(tiling_data.x1_length == tiling_data.x2_length){
    KernelNotEqual<DTYPE_X1> op;
    op.Init(x1, x2, y, tiling_data.totalLength, tiling_data.tileNumMean,
      tiling_data.tileNumEnd, tiling_data.tileLengthMean,
      tiling_data.tileLengthEnd, tiling_data.blockLengthMean,
      tiling_data.blockLengthEnd);
    op.Process();
  }else{
    KernelNotEqual_Broadcast<DTYPE_X1> op;
    op.Init(x1, x2, y, tiling_data.totalLength, tiling_data.tileNumMean,
      tiling_data.tileNumEnd, tiling_data.tileLengthMean,
      tiling_data.tileLengthEnd, tiling_data.blockLengthMean,
      tiling_data.blockLengthEnd, tiling_data.x1_length, tiling_data.x2_length, tiling_data.DimNum, tiling_data.shape, tiling_data.shapefull);
    op.Process();
  }
}
