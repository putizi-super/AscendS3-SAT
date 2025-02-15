#include <type_traits>
#include "kernel_operator.h"
#include <limits> 
#include <cmath> // For std::isnan 

constexpr int32_t BUFFER_NUM = 2;  // tensor num for each queue
constexpr float ZERO_FLOAT = 0.0F;
constexpr float NEGATIVE_ONE_FP32 = -1.0F;
constexpr float POSITIVE_ONE_FP32 = 1.0F;
constexpr int32_t NEGATIVE_ONE_I32 = -1;
constexpr int32_t POSITIVE_ONE_I32 = 1;
constexpr float MIN_ACCURACY_FP16 = 0.00000005960464477539063F;
constexpr float MAX_MUL_FP16 = 4096;
constexpr float MIN_ACCURACY_FP32 = 1.1754943508222875e-38;
constexpr float MAX_MUL_1_FP32 = 1125899906842624;
constexpr float MAX_MUL_2_FP32 = 67108864;
constexpr uint32_t BLOCK_SIZE = 32;

constexpr float NAN_FP32= std::numeric_limits<float>::quiet_NaN();;
constexpr int16_t NAN_INT16= 16128;
constexpr int32_t NAN_INT32= 1071644672;


template <typename typeT>
// global_mem -> unified_buffer  calCount数据个数
// AscendC:DataCopy 按照一次32B处理
__aicore__ inline void DataCopyPadCustom_GM2UB(
  const AscendC::LocalTensor<typeT>& dstLocal, const AscendC::GlobalTensor<typeT>& srcGlobal,
  const uint32_t calCount) {
  if (calCount < BLOCK_SIZE / sizeof(typeT)) {  // 少于32B的数据直接赋值
    for (uint32_t i = 0; i < calCount; i++) {
      dstLocal.SetValue(i, srcGlobal.GetValue(i));
    }
  }
  else {  // 多于32B的数据先将32B的倍数copy，剩下不对齐的再赋值
    uint32_t padDataCount = calCount - (calCount % (BLOCK_SIZE / sizeof(typeT))); // 可以对齐32B的数据个数如 calCount=33 padDataCount=32
    AscendC::DataCopy(dstLocal, srcGlobal, padDataCount); // 拷贝可以对齐的
    for (uint32_t i = 0; i < (calCount % (BLOCK_SIZE / sizeof(typeT))); i++) {
      dstLocal[padDataCount].SetValue(i, srcGlobal[padDataCount].GetValue(i));
    }
  }
}

template <typename typeT>
__aicore__ inline void DataCopyPadCustom_UB2GM(
  const AscendC::GlobalTensor<typeT>& dstGlobal, const AscendC::LocalTensor<typeT>& srcLocal,
  const uint32_t calCount) {
  if (calCount < BLOCK_SIZE / sizeof(typeT)) {
    for (uint32_t i = 0; i < calCount; i++) {
      typeT localValue = srcLocal.GetValue(i);
      auto cursor = dstGlobal.address_ + i;
      *cursor = localValue;
    }
  }
  else {
    uint32_t padDataCount = calCount - (calCount % (BLOCK_SIZE / sizeof(typeT)));
    AscendC::DataCopy(dstGlobal, srcLocal, padDataCount);
    for (uint32_t i = 0; i < (calCount % (BLOCK_SIZE / sizeof(typeT))); i++) {
      typeT localValue = srcLocal[padDataCount].GetValue(i);
      auto cursor = dstGlobal[padDataCount].address_ + i;
      *cursor = localValue;
    }
  }
}

template <typename typeT>
class KernelIsClose {
public:
  __aicore__ inline KernelIsClose() {}
  // 初始化分配空间  Block GM->UB(LocalInput Tensor)的数据块32B
  // total_length 总体数据的总个数
  //                a)所有数据一次移动就可以处理完<32B  b)  
  // tile_num_mean      平均要处理的tile的数量
  // tile_num_end       最后要处理的tile的数量
  // tile_length_mean   平均要处理的tile的数据个数
  // tile_length_end    最后要处理的tile的数据个数
  // block_length_mean  平均的block的数量
  // block_length_end   最后要单独处理的block的数量
  // 举例 
  // Init param total_length[128], tile_num_mean[1],tile_num_end[1], tile_length_mean[128], tile_length_end[128], block_length_mean[128], block_length_end[128]
  //
  __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
    uint32_t total_length, uint32_t tile_num_mean,
    uint32_t tile_num_end, uint32_t tile_length_mean,
    uint32_t tile_length_end,
    uint32_t block_length_mean,
    uint32_t block_length_end,
    float atol,float rtol, bool equalNan) {
    // 接收数据分片
    ResovleTiling(total_length, tile_num_mean, tile_num_end, tile_length_mean,
      tile_length_end, block_length_mean, block_length_end, atol, rtol, equalNan);
    // 分配GM
    x1_gm.SetGlobalBuffer(
      (__gm__ typeT*)x1 + this->block_offset * AscendC::GetBlockIdx(),
      this->block_length);
    x2_gm.SetGlobalBuffer(
      (__gm__ typeT*)x2 + this->block_offset * AscendC::GetBlockIdx(),
      this->block_length);
    y_gm.SetGlobalBuffer((__gm__ int8_t*)y + this->block_offset * AscendC::GetBlockIdx(),
      this->block_length);
    // 分配队列 输入队列x1, x2 输出队列y  临时变量calc_buf_1
    pipe.InitBuffer(x1_inque, BUFFER_NUM, this->tile_cache * sizeof(typeT));
    pipe.InitBuffer(x2_inque, BUFFER_NUM, this->tile_cache * sizeof(typeT));
    pipe.InitBuffer(y_outque, BUFFER_NUM, this->tile_cache * sizeof(int8_t));

    pipe.InitBuffer(calc_buf_1, this->tile_cache * sizeof(typeT));

    pipe.InitBuffer(calc_buf_uint8, this->tile_cache * sizeof(uint8_t));
    pipe.InitBuffer(calc_buf2_uint8, this->tile_cache * sizeof(uint8_t));

    pipe.InitBuffer(calc_buf_int8, this->tile_cache * sizeof(int8_t));
    pipe.InitBuffer(calc_buf2_int8, this->tile_cache * sizeof(int8_t));
    
    pipe.InitBuffer(calc_buf_int16, this->tile_cache * sizeof(int16_t));    
    pipe.InitBuffer(calc_buf2_int16, this->tile_cache * sizeof(int16_t));

    pipe.InitBuffer(calc_buf_half, this->tile_cache * sizeof(half));    
    pipe.InitBuffer(calc_buf2_half, this->tile_cache * sizeof(half));

    pipe.InitBuffer(calc_buf_int32, this->tile_cache * sizeof(int32_t));
    pipe.InitBuffer(calc_buf2_int32, this->tile_cache * sizeof(int32_t));

    pipe.InitBuffer(calc_buf_float,this->tile_cache * sizeof(float));
    pipe.InitBuffer(calc_buf2_float, this->tile_cache * sizeof(float));
    pipe.InitBuffer(calc_buf3_float, this->tile_cache * sizeof(float));

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
    // 处理最后一块使用pad对齐
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
    uint32_t tile_length_mean, uint32_t tile_length_end,
    uint32_t block_length_mean, uint32_t block_length_end,
    float atol,float rtol, bool equalNan) {
    
    this->atol = atol;
    this->rtol = rtol;
    this->equalNan = equalNan;    

    uint32_t pad32 = BLOCK_SIZE / sizeof(typeT);  // 对齐32B需要的最小数据量
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
    DataCopyPadCustom_GM2UB(x1_local, x1_gm[progress * this->tile_cache],
      this->tile_length_end);
    DataCopyPadCustom_GM2UB(x2_local, x2_gm[progress * this->tile_cache],
      this->tile_length_end);
    x1_inque.EnQue(x1_local);
    x2_inque.EnQue(x2_local);
  }

  __aicore__ inline void Calculate(AscendC::LocalTensor<float> &x1_local, AscendC::LocalTensor<float> &x2_local, AscendC::LocalTensor<int8_t> &y_local) {
 
    //  // 比较x1==x1 nan为0 其他元素为 
    //     AscendC::LocalTensor<uint8_t> x1_self_cp_uint8 = calc_buf_uint8.Get<uint8_t>();
    //     AscendC::LocalTensor<uint8_t> x2_self_cp_uint8 = calc_buf2_uint8.Get<uint8_t>();
    //     AscendC::Compare(x1_self_cp_uint8, x1_local, x1_local, AscendC::CMPMODE::EQ, this->tile_cache);
    //     AscendC::Compare(x2_self_cp_uint8, x2_local, x2_local, AscendC::CMPMODE::EQ, this->tile_cache);  
    //     AscendC::LocalTensor<float> y_compute = calc_buf2_float.Get<float>(); 

    //     // 1. y_compute=|x1-x2|
    //     AscendC::Sub(y_compute, x1_local, x2_local, this->tile_cache);   // 第二个参数 - 第三个参数
    //     AscendC::Abs(y_compute, y_compute, this->tile_cache);
    //     // 2. x2=|x2|
    //     AscendC::Abs(x2_local, x2_local, this->tile_cache);
    //               AscendC::printf("自比8 x2  %f\n", x2_local.GetValue(0)); 
    //     // AscendC::Muls(x2_local, x2_local, (float)this->rtol, this->tile_cache); // 第三个参数scalar 要和目的参数保持一致

    //  // 分离等式两侧  整数和小数 x1_num=int32(x1_local)        x1_dot=x1_local-x1_num
    //   //                  cp_num()  < 直接为true    > 直接为false   ==继续比较小数
    //   // 比较小数   小数左移动MAX_MUL_1_FP32位后  再做乘法 小于等于的比较
    //     AscendC::LocalTensor<int32_t> x1_num_int = calc_buf_int32.Get<int32_t>();
    //     AscendC::LocalTensor<int32_t> x2_num_int = calc_buf2_int32.Get<int32_t>();

    //     AscendC::Cast(x1_num_int, y_compute, AscendC::RoundMode::CAST_NONE, this->tile_cache);
    //     AscendC::Cast(x2_num_int, x2_local, AscendC::RoundMode::CAST_NONE, this->tile_cache);
    //             AscendC::printf("自比8 x2  %d\n", x2_num_int.GetValue(0)); 


    //     // 整数部分的浮点数
    //     AscendC::LocalTensor<float> x1_num_float = calc_buf_float.Get<float>();
    //     AscendC::LocalTensor<float> x2_num_float = calc_buf2_float.Get<float>();
    //     AscendC::Cast(x1_num_float, x1_num_int, AscendC::RoundMode::CAST_NONE, this->tile_cache);
    //     AscendC::Cast(x2_num_float, x2_num_int, AscendC::RoundMode::CAST_NONE, this->tile_cache);
    //     AscendC::printf("自比8 x1 %f x2  %f\n", x1_num_float.GetValue(0), x2_num_float.GetValue(0)); 
    //     // 小数部分的浮点数
    //     AscendC::LocalTensor<float> x1_dot_float = calc_buf2_float.Get<float>();
    //     AscendC::LocalTensor<float> x2_dot_float = calc_buf_float.Get<float>();
    //     AscendC::Sub(x1_dot_float, y_compute, x1_num_float, this->tile_cache);
    //     AscendC::Sub(x2_dot_float, x2_local, x2_num_float, this->tile_cache);
    //     AscendC::Muls(x1_dot_float, x1_dot_float, (float)1000000, this->tile_cache);
    //     AscendC::Muls(x2_dot_float, x2_dot_float, (float)1000000, this->tile_cache);
    //     // 此时x2*rtol 分开变
    //     AscendC::Muls(x1_num_float, x1_num_float, (float)this->rtol, this->tile_cache);
    //     AscendC::Muls(x2_dot_float, x2_dot_float, (float)this->rtol, this->tile_cache);
    //     AscendC::Adds(x2_dot_float, x2_dot_float, (float)(this->atol * 1000000), this->tile_cache);
    //     AscendC::printf("自比8 x1 %f x2  %f\n", x1_num_float.GetValue(0), x2_num_float.GetValue(0)); 
    //     AscendC::printf("自比8 x1 %f x2  %f\n", x1_dot_float.GetValue(0), x2_dot_float.GetValue(0)); 
    //     // isclose的标准是 整数部分左侧小于等于右侧   小数部分左侧小于右侧
    //     // 整数部分的比较为最高优先级，小数部分为次高优先级
    //     // 比较整数
    //     AscendC::LocalTensor<uint8_t> num_cp_lt_uint8 = calc_buf_uint8.Get<uint8_t>();
    //     AscendC::LocalTensor<uint8_t> num_cp_eq_uint8 = calc_buf_uint8.Get<uint8_t>();
    //     AscendC::Compare(num_cp_lt_uint8, x1_num_float, x2_num_float, AscendC::CMPMODE::LT, this->tile_cache);
    //     AscendC::Compare(num_cp_eq_uint8, x1_num_float, x2_num_float, AscendC::CMPMODE::EQ, this->tile_cache);
    //     // 比较小数
    //     AscendC::LocalTensor<uint8_t> dot_cp_le_uint8 = calc_buf2_uint8.Get<uint8_t>();

    //     AscendC::Compare(dot_cp_le_uint8, x1_dot_float, x2_dot_float, AscendC::CMPMODE::LE, this->tile_cache); 

    //     AscendC::printf("自比8 x1 %x x2  %x    x3 %x\n", num_cp_lt_uint8.GetValue(0), num_cp_eq_uint8.GetValue(0), dot_cp_le_uint8.GetValue(0)); 
    //     // (整数相等&&小数小于等于)或(整数小于) 暂存到 num_cp_eq_uint8
    //     AscendC::And(num_cp_eq_uint8, num_cp_eq_uint8, dot_cp_le_uint8, this->tile_cache);
    //     AscendC::Or(num_cp_eq_uint8, num_cp_lt_uint8, num_cp_eq_uint8, this->tile_cache);
        
    //     // 定义全0和全1
    //     AscendC::LocalTensor<int8_t> zero_int8 = calc_buf_uint8.Get<int8_t>();
    //     AscendC::LocalTensor<int8_t> one_int8 = calc_buf2_uint8.Get<int8_t>();

    //     AscendC::LocalTensor<half> zero_half = calc_buf_half.Get<half>();
    //     AscendC::LocalTensor<half> one_half = calc_buf2_half.Get<half>();
    //     AscendC::Duplicate(zero_half, (half)0, this->tile_cache);
    //     AscendC::Duplicate(one_half, (half)1, this->tile_cache);
        
    //     AscendC::Cast(zero_int8, zero_half, AscendC::RoundMode::CAST_NONE, this->tile_cache);
    //     AscendC::Cast(one_int8, one_half, AscendC::RoundMode::CAST_NONE, this->tile_cache);


    //     // 使整数和小数比较结果
    //     AscendC::Select(y_local, num_cp_eq_uint8, one_int8, zero_int8, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->tile_cache);

    //     // 对于x1或x2位置元素为nan直接付0
    //     AscendC::Select(y_local, x1_self_cp_uint8, y_local, zero_int8, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->tile_cache);
    //     AscendC::Select(y_local, x2_self_cp_uint8, y_local, zero_int8,AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->tile_cache);

    //     if(this-> equalNan == true){
    //         // 将x1=x2=Nan置为1
    //         AscendC::LocalTensor<half> x1_self_cp_fp16 = calc_buf_half.Get<half>();
    //         AscendC::LocalTensor<half> x2_self_cp_fp16 = calc_buf2_half.Get<half>();
    //         AscendC::LocalTensor<int16_t> x1_self_cp_int16 = calc_buf_int16.Get<int16_t>();
    //         AscendC::LocalTensor<int16_t> x2_self_cp_int16 = calc_buf2_int16.Get<int16_t>();

    //         // AscendC::printf("自比8 x1 %x x2  %x\n", x1_self_cp_uint8.GetValue(0), x2_self_cp_uint8.GetValue(0)); 
    //         AscendC::Cast(x1_self_cp_fp16, x1_self_cp_uint8, AscendC::RoundMode::CAST_NONE, this->tile_cache);
    //         AscendC::Cast(x2_self_cp_fp16, x2_self_cp_uint8, AscendC::RoundMode::CAST_NONE, this->tile_cache);
    //         AscendC::Cast(x1_self_cp_int16, x1_self_cp_fp16, AscendC::RoundMode::CAST_RINT, this->tile_cache);
    //         AscendC::Cast(x2_self_cp_int16, x2_self_cp_fp16, AscendC::RoundMode::CAST_RINT, this->tile_cache);

    //         AscendC::Not(x1_self_cp_int16, x1_self_cp_int16, this->tile_cache);
    //         AscendC::Not(x2_self_cp_int16, x2_self_cp_int16, this->tile_cache);
    //         // AscendC::printf("取反 x1 %x x2  %x\n", x1_self_cp_int16.GetValue(0), x2_self_cp_int16.GetValue(0)); 
    //         AscendC::And(x1_self_cp_int16, x1_self_cp_int16, x2_self_cp_int16, this->tile_cache);
    //         // AscendC::printf("求与  %x \n", x1_self_cp_int16.GetValue(0)); 
    //         AscendC::Not(x1_self_cp_int16, x1_self_cp_int16, this->tile_cache);
    //         AscendC::printf("取反 %x \n", x1_self_cp_int16.GetValue(0));                      

    //         // 使用最后nan==nan的mask
    //         AscendC::Cast(x1_self_cp_fp16, x1_self_cp_int16, AscendC::RoundMode::CAST_NONE, this->tile_cache); 
    //         AscendC::Cast(x1_self_cp_uint8, x1_self_cp_fp16, AscendC::RoundMode::CAST_NONE, this->tile_cache);
    //         AscendC::Select(y_local, x1_self_cp_uint8, y_local,one_int8,  AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->tile_cache); // 此时x1_local全1
    //     }
      // 比较x1==x1 nan为0 其他元素为 
        AscendC::LocalTensor<uint8_t> x1_self_cp_uint8 = calc_buf_uint8.Get<uint8_t>();
        AscendC::LocalTensor<uint8_t> x2_self_cp_uint8 = calc_buf2_uint8.Get<uint8_t>();
        AscendC::Compare(x1_self_cp_uint8, x1_local, x1_local, AscendC::CMPMODE::EQ, this->tile_cache);
        AscendC::Compare(x2_self_cp_uint8, x2_local, x2_local, AscendC::CMPMODE::EQ, this->tile_cache);  
        AscendC::LocalTensor<float> y_compute = calc_buf2_float.Get<float>(); 
        // 1. y_compute=|x1-x2|
        AscendC::Sub(y_compute, x1_local, x2_local, this->tile_cache);   // 第二个参数 - 第三个参数
        AscendC::Abs(y_compute, y_compute, this->tile_cache);
        AscendC::printf("x1-x2 %f \n", y_compute.GetValue(0)); 

        // 2. x2=atol+rtol*|x2|
        AscendC::Abs(x2_local, x2_local, this->tile_cache);
        AscendC::Muls(x2_local, x2_local, (float)this->rtol, this->tile_cache); // 第三个参数scalar 要和目的参数保持一致
        AscendC::printf("rtol*x2 %f \n", x2_local.GetValue(0)); 
        AscendC::Adds(x2_local, x2_local, (float)this->atol, this->tile_cache); // 第三个参数scalar 要和目的参数类型一直
        AscendC::printf("atol+rtol*x2 %f \n", x2_local.GetValue(0)); 
       
        // 3. y_compute = |x1-x2| - ( atol+rtol*|x2| )    
        AscendC::Sub(y_compute, y_compute, x2_local, this->tile_cache);

        AscendC::printf("|x1-x2| - (atol+rtol*|x2|) %f \n", y_compute.GetValue(0));
        // 4. 期望y_compute  <=0 为真  >0 为假        举例差值为 [-1.5 0  1.5] 最终输出为[1 1 0]  
        // 4.1 先求一个相反的 y_compute  <=0  =0    >0  =1 
        AscendC::Mins(y_compute, y_compute, (float)MIN_ACCURACY_FP32, this->tile_cache); //此处min表示大于0小于1的一个最小的数字 [-1.5 0 min]
        AscendC::Maxs(y_compute, y_compute, (float)ZERO_FLOAT, this->tile_cache); // [0 0 min]
        AscendC::Muls(y_compute, y_compute, (float)MAX_MUL_1_FP32, this->tile_cache); // 左移 
        AscendC::Muls(y_compute, y_compute, (float)MAX_MUL_1_FP32, this->tile_cache); // 左移
        AscendC::Muls(y_compute, y_compute, (float)MAX_MUL_2_FP32, this->tile_cache); // 左移若干位 变为[0 0 1]
        // 4.2 (1 - y_compute) 为终值
        AscendC::Duplicate(x1_local, (float)POSITIVE_ONE_FP32, this->tile_cache); // x1 变为全1
        AscendC::Sub(y_compute, x1_local, y_compute, this->tile_cache);
        AscendC::printf("y_compute %f \n", y_compute.GetValue(0)); 
        AscendC::LocalTensor<float> zero_float = calc_buf_float.Get<float>();
        AscendC::Duplicate(zero_float, (float)ZERO_FLOAT, this->tile_cache);
        // 对于含有nan的比较 有可能算完后为1 此时强制设置为0          
        AscendC::Select(y_compute, x1_self_cp_uint8, y_compute, zero_float, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->tile_cache);
        AscendC::Select(y_compute, x2_self_cp_uint8, y_compute,zero_float,AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->tile_cache);

        AscendC::printf("\n用过nan==nan的mask前最终y_compute的值：\n");
        for(int i =0;i<8;i++){
            AscendC::printf("%f \n", y_compute.GetValue(i));      
        }
        if(this-> equalNan == true){
            // 将x1=x2=Nan置为1
            AscendC::LocalTensor<half> x1_self_cp_fp16 = calc_buf_half.Get<half>();
            AscendC::LocalTensor<half> x2_self_cp_fp16 = calc_buf2_half.Get<half>();
            AscendC::LocalTensor<int16_t> x1_self_cp_int16 = calc_buf_int16.Get<int16_t>();
            AscendC::LocalTensor<int16_t> x2_self_cp_int16 = calc_buf2_int16.Get<int16_t>();

            AscendC::printf("自比8 x1 %x x2  %x\n", x1_self_cp_uint8.GetValue(0), x2_self_cp_uint8.GetValue(0)); 
            AscendC::Cast(x1_self_cp_fp16, x1_self_cp_uint8, AscendC::RoundMode::CAST_NONE, this->tile_cache);
            AscendC::Cast(x2_self_cp_fp16, x2_self_cp_uint8, AscendC::RoundMode::CAST_NONE, this->tile_cache);
            AscendC::Cast(x1_self_cp_int16, x1_self_cp_fp16, AscendC::RoundMode::CAST_RINT, this->tile_cache);
            AscendC::Cast(x2_self_cp_int16, x2_self_cp_fp16, AscendC::RoundMode::CAST_RINT, this->tile_cache);

            AscendC::Not(x1_self_cp_int16, x1_self_cp_int16, this->tile_cache);
            AscendC::Not(x2_self_cp_int16, x2_self_cp_int16, this->tile_cache);
            AscendC::printf("取反 x1 %x x2  %x\n", x1_self_cp_int16.GetValue(0), x2_self_cp_int16.GetValue(0)); 
            AscendC::And(x1_self_cp_int16, x1_self_cp_int16, x2_self_cp_int16, this->tile_cache);
            AscendC::printf("求与  %x \n", x1_self_cp_int16.GetValue(0)); 
            AscendC::Not(x1_self_cp_int16, x1_self_cp_int16, this->tile_cache);
            AscendC::printf("取反 %x \n", x1_self_cp_int16.GetValue(0));                      

            // 使用最后nan==nan的mask
            AscendC::Cast(x1_self_cp_fp16, x1_self_cp_int16, AscendC::RoundMode::CAST_NONE, this->tile_cache); 
            AscendC::Cast(x1_self_cp_uint8, x1_self_cp_fp16, AscendC::RoundMode::CAST_NONE, this->tile_cache);
            AscendC::Select(y_compute, x1_self_cp_uint8, y_compute,x1_local,  AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->tile_cache);// 此时x1_local全1
            
            AscendC::printf("\n用过nan==nan的mask后最终y_compute的值：\n");
            for(int i =0;i<8;i++){
                AscendC::printf("%f \n", y_compute.GetValue(i));      
            }
        }
        // TODO转换两次的目的? float32->half->int8 最终结果为 [0 0 1]
        AscendC::LocalTensor<half> x1_self_cp_fp16 = calc_buf_half.Get<half>();
        AscendC::Cast(x1_self_cp_fp16, y_compute, AscendC::RoundMode::CAST_NONE, this->tile_cache);
        AscendC::Cast(y_local, x1_self_cp_fp16, AscendC::RoundMode::CAST_NONE, this->tile_cache);
  }

  __aicore__ inline void Compute(int32_t progress) {
    AscendC::LocalTensor<typeT> x1_local = x1_inque.DeQue<typeT>();
    AscendC::LocalTensor<typeT> x2_local = x2_inque.DeQue<typeT>();
    AscendC::LocalTensor<int8_t> y_local = y_outque.AllocTensor<int8_t>();
    if constexpr (std::is_same_v<typeT, half>) {
        AscendC::LocalTensor<float> x1_fp32 = calc_buf2_float.Get<float>();
        AscendC::LocalTensor<float> x2_fp32 = calc_buf_float.Get<float>();        
        AscendC::Cast(x1_fp32, x1_local, AscendC::RoundMode::CAST_NONE, this->tile_cache);
        AscendC::Cast(x2_fp32, x2_local, AscendC::RoundMode::CAST_NONE, this->tile_cache);
        Calculate(x1_fp32, x2_fp32, y_local);
    //   if(this-> equalNan == false){
    //     // 比较x1==x1 nan为0 其他元素为1
    //     AscendC::LocalTensor<uint8_t> x1_self_cp_uint8 = calc_buf_uint8.Get<uint8_t>();
    //     AscendC::LocalTensor<uint8_t> x2_self_cp_uint8 = calc_buf2_uint8.Get<uint8_t>();
    //     AscendC::Compare(x1_self_cp_uint8, x1_local, x1_local, AscendC::CMPMODE::EQ, this->tile_cache);
    //     AscendC::Compare(x2_self_cp_uint8, x2_local, x2_local, AscendC::CMPMODE::EQ, this->tile_cache);  


    //     AscendC::LocalTensor<typeT> y_compute = calc_buf_1.Get<typeT>(); 
    //     // 1. y_compute=|x1-x2|
    //     AscendC::Sub(y_compute, x1_local, x2_local, this->tile_cache);   // 第二个参数 - 第三个参数
    //     AscendC::Abs(y_compute, y_compute, this->tile_cache);
    //     // 2. x2=atol+rtol*|x2|
    //     AscendC::Abs(x2_local, x2_local, this->tile_cache);
    //     AscendC::Muls(x2_local, x2_local, (half)this->rtol, this->tile_cache); // 第三个参数scalar 要和目的参数保持一致
    //     AscendC::Adds(x2_local, x2_local, (half)this->atol, this->tile_cache); // 第三个参数scalar 要和目的参数类型一直  
    //     // 3. y_compute = |x1-x2|    -    ( atol+rtol*|x2| )    
    //     AscendC::Sub(y_compute, y_compute, x2_local, this->tile_cache);
    //     // 4. y_compute  <=0 为真  >0 为假        举例差值为 [-1.5 0  1.5] 最终输出为[1 1 0]  
    //     // 4.1 y_compute  <=0  =0    >0  =1
    //     AscendC::Mins(y_compute, y_compute, (half)MIN_ACCURACY_FP16, this->tile_cache); //此处min表示大于0小于1的一个数字 [-1.5 0 min]
    //     AscendC::Maxs(y_compute, y_compute, (half)ZERO_FLOAT, this->tile_cache); // [0 0 min]
    //     AscendC::Muls(y_compute, y_compute, (half)MAX_MUL_FP16, this->tile_cache); // 左移4096 12位
    //     AscendC::Muls(y_compute, y_compute, (half)MAX_MUL_FP16, this->tile_cache); // 左移4096  变为 [0 0 1]
    //     // 4.2 1 - y_compute 为终值
    //     AscendC::Duplicate(x1_local, (half)POSITIVE_ONE_FP32, this->tile_cache); // x1 变为全1
    //     AscendC::Sub(y_compute, x1_local, y_compute, this->tile_cache);

    //     // 对于含有nan的比较 有可能算完后为1 此时强制设置为0          
    //     AscendC::Duplicate(x2_local, (half)ZERO_FLOAT, this->tile_cache);
    //     AscendC::printf("自比8 x1 %x \n", x1_self_cp_uint8.GetValue(0));
    //     AscendC::Select(y_compute, x1_self_cp_uint8, y_compute, x2_local, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->tile_cache);
    //     AscendC::Select(y_compute, x2_self_cp_uint8, y_compute,x2_local, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->tile_cache);



    //     AscendC::Cast(y_local, y_compute, AscendC::RoundMode::CAST_NONE, this->tile_cache); //最终结果为 [0 0 1]
    //   }else{
    //     AscendC::LocalTensor<uint8_t> x1_self_cp_uint8 = calc_buf_uint8.Get<uint8_t>();
    //     AscendC::LocalTensor<uint8_t> x2_self_cp_uint8 = calc_buf2_uint8.Get<uint8_t>();
    //     AscendC::LocalTensor<half> x1_self_cp_fp16 = calc_buf_half.Get<half>();
    //     AscendC::LocalTensor<half> x2_self_cp_fp16 = calc_buf2_half.Get<half>();
    //     AscendC::LocalTensor<int16_t> x1_self_cp_int16 = calc_buf_int16.Get<int16_t>();
    //     AscendC::LocalTensor<int16_t> x2_self_cp_int16 = calc_buf2_int16.Get<int16_t>();
                


    //     AscendC::Compare(x1_self_cp_uint8, x1_local, x1_local, AscendC::CMPMODE::EQ, this->tile_cache);
    //     AscendC::Compare(x2_self_cp_uint8, x2_local, x2_local, AscendC::CMPMODE::EQ, this->tile_cache);
    //     AscendC::printf("自比8 x1 %x x2  %x\n", x1_self_cp_uint8.GetValue(0), x2_self_cp_uint8.GetValue(0)); 
    //     AscendC::Cast(x1_self_cp_fp16, x1_self_cp_uint8, AscendC::RoundMode::CAST_NONE, this->tile_cache);
    //     AscendC::Cast(x2_self_cp_fp16, x2_self_cp_uint8, AscendC::RoundMode::CAST_NONE, this->tile_cache);
    //     AscendC::printf("自比fp16 x1 %f x2  %f\n", x1_self_cp_fp16.GetValue(0), x2_self_cp_fp16.GetValue(0)); 
    //     AscendC::Cast(x1_self_cp_int16, x1_self_cp_fp16, AscendC::RoundMode::CAST_RINT, this->tile_cache);
    //     AscendC::Cast(x2_self_cp_int16, x2_self_cp_fp16, AscendC::RoundMode::CAST_RINT, this->tile_cache);
    //     AscendC::printf("自比int16 x1 %d x2  %d\n", x1_self_cp_int16.GetValue(0), x2_self_cp_int16.GetValue(0)); 

    //     AscendC::Not(x1_self_cp_int16, x1_self_cp_int16, this->tile_cache);
    //     AscendC::Not(x2_self_cp_int16, x2_self_cp_int16, this->tile_cache);
    //     AscendC::printf("取反 x1 %x x2  %x\n", x1_self_cp_int16.GetValue(0), x2_self_cp_int16.GetValue(0)); 
    //     AscendC::And(x1_self_cp_int16, x1_self_cp_int16, x2_self_cp_int16, this->tile_cache);
    //     AscendC::printf("求与  %x \n", x1_self_cp_int16.GetValue(0)); 
    //     AscendC::Not(x1_self_cp_int16, x1_self_cp_int16, this->tile_cache);
    //     AscendC::printf("取反 %x \n", x1_self_cp_int16.GetValue(0));                      
        
   
    //     // 原始isclose的过程 
    //     AscendC::LocalTensor<typeT> y_compute = calc_buf_1.Get<typeT>(); 
    //     // 1. y_compute=|x1-x2|
    //     AscendC::Sub(y_compute, x1_local, x2_local, this->tile_cache);   // 第二个参数 - 第三个参数
    //     AscendC::Abs(y_compute, y_compute, this->tile_cache);
    //     // 2. x2=atol+rtol*|x2|
    //     AscendC::Abs(x2_local, x2_local, this->tile_cache);
    //     AscendC::Muls(x2_local, x2_local, (half)this->rtol, this->tile_cache); // 第三个参数scalar 要和目的参数保持一致
    //     AscendC::Adds(x2_local, x2_local, (half)this->atol, this->tile_cache); // 第三个参数scalar 要和目的参数类型一直  
        
    //     // 3. y_compute = |x1-x2|    -    ( atol+rtol*|x2| )    
    //     AscendC::Sub(y_compute, y_compute, x2_local, this->tile_cache);
    //     // AscendC::printf("区间映射前y_compute的值：");
    //     // for(int i =0;i<8;i++){
    //     //     AscendC::printf("%f ", y_compute.GetValue(i));      
    //     // }

        
    //     // 4. y_compute  <=0 为真  >0 为假        举例差值为 [-1.5 0  1.5] 最终输出为[1 1 0]  
    //     // 4.1 y_compute  <=0  =0    >0  =1
    //     AscendC::Mins(y_compute, y_compute, (half)MIN_ACCURACY_FP16, this->tile_cache); //此处min表示大于0小于1的一个数字 [-1.5 0 min]
    //     AscendC::Maxs(y_compute, y_compute, (half)ZERO_FLOAT, this->tile_cache); // [0 0 min]
    //     AscendC::Muls(y_compute, y_compute, (half)MAX_MUL_FP16, this->tile_cache); // 左移4096 12位
    //     AscendC::Muls(y_compute, y_compute, (half)MAX_MUL_FP16, this->tile_cache); // 左移4096  变为 [0 0 1]
    //     // 4.2 1 - y_compute 为终值
    //     AscendC::Duplicate(x1_local, (half)POSITIVE_ONE_FP32, this->tile_cache); // x1 变为全1
    //     AscendC::Sub(y_compute, x1_local, y_compute, this->tile_cache);

    //     // 由于nan导致y_compute真值为1了，所以我们使用x1_self_cp_uint8、x2_self_cp_uint8 把对应含nan的位置变为0
    //     AscendC::Duplicate(x2_local, (half)ZERO_FLOAT, this->tile_cache);
    //     AscendC::printf("自比8 x1 %x \n", x1_self_cp_uint8.GetValue(0));
    //     AscendC::Select(y_compute, x1_self_cp_uint8, y_compute, x2_local, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->tile_cache);
    //     AscendC::Select(y_compute, x2_self_cp_uint8, y_compute,x2_local, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->tile_cache);


    //     // 使用nan的mask进行 select过程 select支持half
    //     AscendC::printf("\n最终y_compute的值：");
    //     for(int i =0;i<8;i++){
    //         AscendC::printf("%f ", y_compute.GetValue(i));      
    //     }


    //     // 使用最后nan==nan的mask
    //     AscendC::Cast(x1_self_cp_fp16, x1_self_cp_int16, AscendC::RoundMode::CAST_NONE, this->tile_cache);
    //     AscendC::printf("mask_fp16 %f \n", x1_self_cp_fp16.GetValue(0)); 
    //     AscendC::Cast(x1_self_cp_uint8, x1_self_cp_fp16, AscendC::RoundMode::CAST_NONE, this->tile_cache);
    //     AscendC::printf("mask_uint8 %x \n", x1_self_cp_uint8.GetValue(0)); 
    //     AscendC::Select(y_compute, x1_self_cp_uint8, y_compute,x1_local,  AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->tile_cache);
        
    //     AscendC::printf("\n用过nan==nan的mask后最终y_compute的值：");
    //     for(int i =0;i<8;i++){
    //         AscendC::printf("%f ", y_compute.GetValue(i));      
    //     }


    //     AscendC::Cast(y_local, y_compute, AscendC::RoundMode::CAST_RINT, this->tile_cache); //最终结果为 [0 0 1]

    //     // 最后的y_compute，全1矩阵 使用我们提供的mask 选择需要做一个select 

    //     // 对于np.nan来说是否返回1
    //     // AscendC::CompareScalar(y_local, x1_int16, NAN_INT16, AscendC::CMPMODE::EQ, this->tile_cache);
    //   }
    }
    else if constexpr (std::is_same_v<typeT, float>) {
        Calculate(x1_local,x2_local, y_local);
    }
    else if constexpr (std::is_same_v<typeT, int32_t>) {      
        AscendC::LocalTensor<float> x1_fp32 = calc_buf2_float.Get<float>();
        AscendC::LocalTensor<float> x2_fp32 = calc_buf_float.Get<float>();
        AscendC::Cast(x1_fp32, x1_local, AscendC::RoundMode::CAST_NONE, this->tile_cache);
        AscendC::Cast(x2_fp32, x2_local, AscendC::RoundMode::CAST_NONE, this->tile_cache);
        Calculate(x1_fp32, x2_fp32, y_local);


    //   AscendC::LocalTensor<float> x1_fp32 = calc_buf2_float.Get<float>();
    //   AscendC::LocalTensor<float> x2_fp32 = calc_buf_float.Get<float>();
    //   AscendC::LocalTensor<float> y_compute_fp32 = calc_buf2_float.Get<float>();
      
    //   AscendC::Cast(x1_fp32, x1_local, AscendC::RoundMode::CAST_NONE, this->tile_cache);
    //   AscendC::Cast(x2_fp32, x2_local, AscendC::RoundMode::CAST_NONE, this->tile_cache);
     
    //    // 1. y_compute_fp32=|x1-x2|
    //   AscendC::Sub(y_compute_fp32, x1_fp32, x2_fp32, this->tile_cache);   // 第二个参数 - 第三个参数
    //   AscendC::Abs(y_compute_fp32, y_compute_fp32, this->tile_cache);
    //   // 2. x2=atol+rtol*|x2|
    //   AscendC::Abs(x2_fp32, x2_fp32, this->tile_cache);
    //   AscendC::Muls(x2_fp32, x2_fp32, (float)this->rtol, this->tile_cache); // 第三个参数scalar 要和目的参数保持一致x
    //   AscendC::Adds(x2_fp32, x2_fp32, (float)this->atol, this->tile_cache); // 第三个参数scalar 要和目的参数类型一直  
    //   // 3. y_compute_fp32 = |x1-x2|    -    ( atol+rtol*|x2| )    
    //   AscendC::Sub(y_compute_fp32, y_compute_fp32, x2_fp32, this->tile_cache);
    //   // 4. 期望y_compute_fp32  <=0 为真  >0 为假        举例差值为 [-1.5 0  1.5] 最终输出为[1 1 0]  
    //   // 4.1 先求一个相反的 y_compute_fp32  <=0  =0    >0  =1 
    //   AscendC::Mins(y_compute_fp32, y_compute_fp32, (float)MIN_ACCURACY_FP32, this->tile_cache); //此处min表示大于0小于1的一个最小的数字 [-1.5 0 min]
    //   AscendC::Maxs(y_compute_fp32, y_compute_fp32, (float)ZERO_FLOAT, this->tile_cache); // [0 0 min]
    //   AscendC::Muls(y_compute_fp32, y_compute_fp32, (float)MAX_MUL_1_FP32, this->tile_cache); // 左移 
    //   AscendC::Muls(y_compute_fp32, y_compute_fp32, (float)MAX_MUL_1_FP32, this->tile_cache); // 左移
    //   AscendC::Muls(y_compute_fp32, y_compute_fp32, (float)MAX_MUL_2_FP32, this->tile_cache); // 左移若干位 变为[0 0 1]
    //   // 4.2 (1 - y_compute_fp32) 为终值
    //   AscendC::Duplicate(x1_fp32, (float)POSITIVE_ONE_FP32, this->tile_cache); // x1 变为全1
    //   AscendC::Sub(y_compute_fp32, x1_fp32, y_compute_fp32, this->tile_cache);
    //   // TODO转换两次的目的? float32->half->int8 最终结果为 [0 0 1]
    //   AscendC::LocalTensor<half> y_fp16 = calc_buf_half.Get<half>();
    //   AscendC::Cast(y_fp16, y_compute_fp32, AscendC::RoundMode::CAST_NONE, this->tile_cache);
    //   AscendC::Cast(y_local, y_fp16, AscendC::RoundMode::CAST_NONE, this->tile_cache);
    }
    else if constexpr (std::is_same_v<typeT, uint8_t>) {
        AscendC::LocalTensor<half> x1_fp16 = calc_buf_half.Get<half>();
        AscendC::LocalTensor<half> x2_fp16 = calc_buf2_half.Get<half>();
        AscendC::Cast(x1_fp16, x1_local, AscendC::RoundMode::CAST_NONE, this->tile_cache);
        AscendC::Cast(x2_fp16, x2_local, AscendC::RoundMode::CAST_NONE, this->tile_cache);

        AscendC::LocalTensor<float> x1_fp32 = calc_buf2_float.Get<float>();
        AscendC::LocalTensor<float> x2_fp32 = calc_buf_float.Get<float>();
        AscendC::Cast(x1_fp32, x1_fp16, AscendC::RoundMode::CAST_NONE, this->tile_cache);
        AscendC::Cast(x2_fp32, x2_fp16, AscendC::RoundMode::CAST_NONE, this->tile_cache);

        Calculate(x1_fp32, x2_fp32, y_local);
    }

    y_outque.EnQue<int8_t>(y_local);
    x1_inque.FreeTensor(x1_local);
    x2_inque.FreeTensor(x2_local);
  }
  __aicore__ inline void CopyOut(int32_t progress) {
    AscendC::LocalTensor<int8_t> y_local = y_outque.DeQue<int8_t>();
    DataCopyPadCustom_UB2GM(y_gm[progress * this->tile_cache], y_local,
      this->tile_cache);
    y_outque.FreeTensor(y_local);
  }
  __aicore__ inline void CopyOutPad(int32_t progress) {
    AscendC::LocalTensor<int8_t> y_local = y_outque.DeQue<int8_t>();
    DataCopyPadCustom_UB2GM(y_gm[progress * this->tile_cache], y_local,
      this->tile_length_end);
    y_outque.FreeTensor(y_local);
  }

private:
    //AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuf, calcBuf1, calcBuf2; // 缓冲变量 计算需要临时变量时使用此位置。
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf_1,calc_buf_uint8,calc_buf2_uint8,calc_buf_int8,calc_buf2_int8,calc_buf_half,calc_buf2_half,calc_buf_int16,calc_buf2_int16,calc_buf_int32,calc_buf2_int32,calc_buf_float,calc_buf2_float,calc_buf3_float;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> x1_inque, x2_inque;
  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> y_outque;
  AscendC::GlobalTensor<typeT> x1_gm, x2_gm;
  AscendC::GlobalTensor<int8_t> y_gm;
  uint32_t total_length; // 所有数据长度
  uint32_t block_length; // 单核要处理的长度
  uint32_t block_offset; // 单核的偏移地址
  uint32_t tile_num; // 单核内多路并行数量
  uint32_t tile_cache; // 单核内实际一路要处理的数量，前面若干路是32B，最后一路要指定
  uint32_t tile_length; // 单核内一路要处理的长度
  uint32_t tile_length_end; // 
  float atol;
  float rtol;
  bool equalNan;
};

extern "C" __global__ __aicore__ void is_close(GM_ADDR x1, GM_ADDR x2,
  GM_ADDR y,
  GM_ADDR workspace,
  GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  KernelIsClose<DTYPE_X1> op;
  op.Init(x1, x2, y, tiling_data.totalLength, tiling_data.tileNumMean,
    tiling_data.tileNumEnd, tiling_data.tileLengthMean,
    tiling_data.tileLengthEnd, tiling_data.blockLengthMean,
    tiling_data.blockLengthEnd, tiling_data.atol,
    tiling_data.rtol, tiling_data.equalNan);
  op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
// call of kernel function
void is_close_do(uint32_t blockDim, void* l2ctrl, void* stream,
  uint8_t* x1, uint8_t* x2, uint8_t* y, uint8_t* workspace,
  uint8_t* tiling) {
  is_close << <blockDim, l2ctrl, stream >> > (x1, x2, y, workspace, tiling);
}
#endif
