#include "kernel_operator.h"
constexpr uint32_t REDUCE_SUM_ONE_REPEAT = 256;
constexpr uint32_t EACH_BLOCK_SIZE = 32;
constexpr uint32_t BUFFER_NUM = 1;

// 一个block多少字节
#define BLOCK_BYTE 32
// 一个block多少个数据
#define BLOCK_DATA_NUM(dtype) ((BLOCK_BYTE) / (sizeof(dtype)))
// 将data_length扩展到恰好可以被target_aline整除
#define ALINE(data_length, target_aline) ((((data_length)+(target_aline) - 1) / (target_aline)) * (target_aline))
// 将data_length个数据扩展到恰好占满pad_aline字节的整数倍，还需要填充多少个字节
#define PAD_BYTE(data_length, pad_aline, dtype) ((pad_aline) - (((data_length)*sizeof(dtype)) % BLOCK_BYTE))
// 将data_length个数据扩展到恰好占满pad_aline字节的整数倍，还需要填充多少个数据
#define PAD_NUM(data_length, pad_aline, dtype) (((pad_aline) / sizeof(dtype) - ((((data_length)*(sizeof(dtype))) % (pad_aline)) / sizeof(dtype))) % ((pad_aline) / sizeof(dtype)))
// 获取这些数据占了多少个block，不足一个block的算一个block
#define BLOCK_NUM(data_length, block_byte, dtype) (((data_length)*(sizeof(dtype))+(block_byte) -1)/block_byte)
enum Condition {FIRST=0, MIDDLE, LAST};

class KernelSoftmax {
 public:
  __aicore__ inline KernelSoftmax() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                              uint32_t totalLength,
                              uint32_t dimNum,
                              uint32_t firstDim,
                              uint32_t middleDim,
                              uint32_t lastDim,
                              uint32_t dim,
                              uint32_t tileNum,
                              uint32_t condition,
                              uint32_t dtype) {
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    ASSERT(tileNum != 0 && "tileNum can not be zero!");
    this->totalLength = totalLength;
    this->dimNum = dimNum;
    this->firstDim = firstDim;
    this->middleDim = middleDim;
    // lastDim表示数据的最后一个维度，目前只支持默认的对最后一个维度做softmax
    this->lastDim = lastDim;
    this->dim = dim;
    this->tileNum = tileNum;
    this->condition = condition;
    // AscendC::printf("input_firstDim:%d\n", this->firstDim);
    // AscendC::printf("input_middleDim:%d\n", this->middleDim);
    // AscendC::printf("input_lastDim:%d\n", this->lastDim);
    // 将整体数据分配到多个AICore上，一个AICore一个block
    // 如何划分数据需要根据condition来决定
    /*
    condition == FIRST:
              划分lastDim
    condition == MIDDLE:
              划分firstDim(不一定是最优的，比如firstDim很小，可能会导致有些核没有数据可以处理) TODO:
    condition == LAST:
              划分firstDim
    */
    uint32_t AI_core_num  = AscendC::GetBlockNum();
    // AscendC::printf("AI_core_num:%d\n", AI_core_num);
    uint32_t core_last_dim_block_num = 0;
    uint32_t former_core_num = 0;
    uint32_t core_firstDim = firstDim;
    uint32_t core_middleDim = middleDim;
    uint32_t core_lastDim = lastDim;
    // 最后一个tile使用，因为其他的都是按照block对齐的，最后一个tile会因为处理非对齐数据而特殊处理
    uint32_t core_tail_lastDim = 0;

    uint32_t last_dim_block_num = BLOCK_NUM(this->lastDim, BLOCK_BYTE, float);
    // AscendC::printf("last_dim_block_num: %d\n", last_dim_block_num);
    uint32_t block_alined_last_dim = ALINE(this->lastDim, BLOCK_DATA_NUM(float));
    uint32_t last_core = AI_core_num;

    // 根据AI Core划分block
    switch (condition) {  
        case Condition::FIRST: {  
            former_core_num = last_dim_block_num % AI_core_num;
            core_firstDim = firstDim;
            // core_middleDim这里仅作初始化，该case下并不会被使用
            core_middleDim = middleDim;
            core_last_dim_block_num = AscendC::GetBlockIdx() < former_core_num ? (last_dim_block_num / AI_core_num + 1) : (last_dim_block_num / AI_core_num);
            core_lastDim = core_last_dim_block_num * BLOCK_DATA_NUM(float);
            if(AscendC::GetBlockIdx() == (AI_core_num - 1) && core_lastDim != 0){
                core_tail_lastDim = lastDim - (last_dim_block_num - core_last_dim_block_num)* BLOCK_DATA_NUM(float);
                last_core = (AI_core_num - 1);
            } else if((last_dim_block_num / AI_core_num) == 0 && AscendC::GetBlockIdx() == (former_core_num-1)){
                core_tail_lastDim = lastDim % BLOCK_DATA_NUM(float);
                last_core = former_core_num-1;
                // AscendC::printf("hello_core_tail_lastDim : %d\n", core_tail_lastDim);
            }
            break;  
        }  
        case Condition::MIDDLE: {  
            former_core_num = firstDim % AI_core_num;
            core_firstDim = AscendC::GetBlockIdx() < former_core_num ? (firstDim / AI_core_num + 1) : (firstDim / AI_core_num); 
            core_middleDim = middleDim;
            core_lastDim = lastDim;
            break;  
        }  
        case Condition::LAST: {  
            former_core_num = firstDim % AI_core_num;
            core_firstDim = AscendC::GetBlockIdx() < former_core_num ? (firstDim / AI_core_num + 1) : (firstDim / AI_core_num); 
            core_middleDim = middleDim;
            core_lastDim = lastDim;
            break;  
        }  
        default: {  
            ASSERT(0 && "Undifined Condition Case!");  
            break;  
        }  
    }

    this->AI_core_num = AI_core_num;
    this->core_last_dim_block_num = core_last_dim_block_num;
    this->former_core_num = former_core_num;
    this->core_firstDim = core_firstDim;
    this->core_middleDim = core_middleDim;
    this->core_lastDim = core_lastDim;
    this->core_tail_lastDim = core_tail_lastDim;
    this->last_core = last_core;
    
    uint32_t tiled_batch = firstDim;
    uint32_t tiled_row_length = lastDim;
    uint32_t tiled_col_length = firstDim;
    uint32_t tail_tiled_batch = 0;
    uint32_t tail_tiled_row_length = 0;
    uint32_t tail_tiled_col_length = 0;

    // 根据tileNum划分每个block
    bool have_tail = false;
    bool just_tail = false;
    uint32_t buffer_data_num;
    uint32_t former_tile = 0;
    uint32_t loop_count = tileNum;
    switch (condition) {  
        case Condition::FIRST: {  
            tiled_row_length = (core_last_dim_block_num / tileNum)*BLOCK_DATA_NUM(float);  
            former_tile = core_last_dim_block_num % tileNum;
            loop_count = tiled_row_length == 0 ? former_tile : tileNum;
            // if(AscendC::GetBlockIdx() == (AI_core_num - 1) && core_lastDim != 0){
            //     tail_tiled_row_length = core_tail_lastDim - tiled_row_length*tileNum;
            //     // if(core_last_dim_block_num == 1 && )
            // } else if((last_dim_block_num / AI_core_num) == 0 && AscendC::GetBlockIdx() == (former_core_num-1)){
            //     tail_tiled_row_length = core_tail_lastDim;
            //     AscendC::printf("hello_tiled_row_length : %d\n", tiled_row_length);
            //     AscendC::printf("hello_tail_tiled_row_length : %d\n", tail_tiled_row_length);
            // } else {
            //     tail_tiled_row_length = (core_last_dim_block_num % tileNum)*BLOCK_DATA_NUM(float);
            // }
              
            // if (tail_tiled_row_length!=0) have_tail = true;
            buffer_data_num = (former_tile == 0 ? tiled_row_length : (tiled_row_length+BLOCK_DATA_NUM(float)))*core_firstDim;
            break;  
        }  
        case Condition::MIDDLE: {  
            tiled_batch = core_firstDim / tileNum;  
            former_tile = core_firstDim % tileNum;
            loop_count = tiled_batch == 0 ? former_tile : tileNum;
            tiled_col_length = middleDim;
            // tail_tiled_batch = core_firstDim - tiled_batch * tileNum;  
            // if(tail_tiled_batch!=0) have_tail = true;
            buffer_data_num = (former_tile == 0 ? tiled_batch : (tiled_batch + 1))*core_middleDim*block_alined_last_dim;
            break;  
        }  
        case Condition::LAST: {  
            tiled_col_length = core_firstDim / tileNum;  
            former_tile = core_firstDim % tileNum;
            loop_count = tiled_col_length == 0 ? former_tile : tileNum;
            // tail_tiled_col_length = core_firstDim - tiled_col_length * tileNum;  
            // if(tail_tiled_col_length!=0) have_tail = true;
            buffer_data_num = (former_tile == 0 ? tiled_col_length : (tiled_col_length + 1))*block_alined_last_dim;
            break;  
        }  
        default: {  
            ASSERT(0 && "Undifined Condition Case!");  
            break;  
        }  
    }
    this->tiled_batch = tiled_batch;
    this->tiled_row_length = tiled_row_length;
    this->tiled_col_length = tiled_col_length;
    this->tail_tiled_batch = tail_tiled_batch;
    this->tail_tiled_row_length = tail_tiled_row_length;
    this->tail_tiled_col_length = tail_tiled_col_length;
    this->block_alined_last_dim = block_alined_last_dim;
    this->have_tail = have_tail;
    uint32_t start_position = 0;
    this->former_tile = former_tile;
    this->loop_count = loop_count;

    // 设置最初的Global Tensor起始位置以及大小
    switch (condition) {  
        case Condition::FIRST: {  
            start_position = (
                                  (AscendC::GetBlockIdx() < former_core_num) ? 
                                  (AscendC::GetBlockIdx()*core_lastDim) : 
                                  (former_core_num*(BLOCK_DATA_NUM(float)+core_lastDim) + (AscendC::GetBlockIdx() - former_core_num)*core_lastDim)
                             );
            // 从当前核的start_position到全局数据的最后一个元素
            this->blockLength =  this->totalLength - start_position;
            break;  
        }  
        case Condition::MIDDLE: {  
            start_position = (
                                  (AscendC::GetBlockIdx() < former_core_num) ? 
                                  (AscendC::GetBlockIdx()*core_firstDim*core_middleDim*core_lastDim) : 
                                  (former_core_num*(core_firstDim + 1) + (AscendC::GetBlockIdx() - former_core_num)*core_firstDim)*core_middleDim*core_lastDim
                             );
            this->blockLength = core_firstDim * core_middleDim * core_lastDim;  
            break;  
        }  
        case Condition::LAST: {  
            start_position = (
                                  (AscendC::GetBlockIdx() < former_core_num) ? 
                                  (AscendC::GetBlockIdx()*core_firstDim*core_lastDim) : 
                                  (former_core_num*(core_firstDim + 1) + (AscendC::GetBlockIdx() - former_core_num)*core_firstDim)*core_lastDim
                             );
            this->blockLength = core_firstDim * core_lastDim;   
            break;  
        }  
        default: {  
            ASSERT(0 && "Undifined Condition Case!");  
            break;  
        }  
    }

    // 设置Global Tensor
    xGm.SetGlobalBuffer((__gm__ float *)x + start_position, this->blockLength);
    yGm.SetGlobalBuffer((__gm__ float *)y + start_position, this->blockLength);

    
    // 关闭双缓冲，每个till分到一个block的一部分
    // tilllength是每次处理的数据量
    this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;
    this->outputTileLength = this->outputLength / this->tileNum / BUFFER_NUM;
    // 这里的4表示一个float32占4字节，this->lastDim * 4表示最后一个维度的数据占多少个字节，除以EACH_BLOCK_SIZE后表示占多少个block
    this->blockReduceStride = this->lastDim * 4 / EACH_BLOCK_SIZE;
    // 256/4 = 64，矢量处理函数一次最多处理256Byte，这里的4表示32bit数据的字节数，来算出来mask的值
    // normalMask表示正常情况下一次处理多少个数据，一次repeat
    this->normalMask = REDUCE_SUM_ONE_REPEAT / 4;

    // RepeatTimes表示针对一个tile需要多少次循环才能完全处理
    this->RepeatTimes = this->tileLength / this->lastDim;
    // AscendC::printf("repeattimes: %d \n", this->RepeatTimes);
    // 如果最后一个维度大于32则不填充
    this->padTilingLength = this->tileLength * (32 / this->lastDim);

    uint32_t padded_row_length = ALINE(this->lastDim, BLOCK_DATA_NUM(float));
    uint32_t tile_rows = this->RepeatTimes;
    this-> padded_row_length = padded_row_length;
    // this->tile_rows = tile_rows;
    
    this->buffer_data_num = buffer_data_num;
    /*
      |<------------>|<-->|
      |<------------>|<-->|
      |<------------>|<-->|
      |<------------>|<-->|
      |<------------>|<-->|
          raw_data    pad
    */
    pipe.InitBuffer(inQueueX, BUFFER_NUM, buffer_data_num * sizeof(float));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, buffer_data_num * sizeof(float));
    // 用来暂时保留每一份最后一个dim的reduce结果
    pipe.InitBuffer(reduceRes, RepeatTimes * sizeof(float));

    // 这俩不知道什么作用
    // pipe.InitBuffer(workLocal, this->padTilingLength * 2 * sizeof(float));
    // pipe.InitBuffer(topKIndices, this->lastDim * 3 * sizeof(uint32_t));

  }
  __aicore__ inline void Process() {
    // 获取当前AICore的逻辑ID
    if (AscendC::GetBlockIdx() >= AscendC::GetBlockNum()) {
      return;
    }

    // int32_t loopCount = this->tileNum;

    for (int32_t i = 0; i < this->loop_count; i++) {
      CopyIn(i);
      Compute(i);
      CopyOut(i);
    }
    // if(this->have_tail){
    //   CopyIn(loopCount);
    //   Compute(loopCount);
    //   CopyOut(loopCount);
    // }
    
  }

 private:
  __aicore__ inline void CopyIn(int32_t progress) {
    // progress表示当前进行到第几个tile
    // 首先从inQueueX中申请芯片内部的Local Buffer
    uint32_t progress_lastdim_length = 0;
    switch (this->condition) {  
        case Condition::FIRST: {  
            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            // 然后将存外的Global Memory中将本次需要处理的tile数据copy到内部的Local Buffer中
            progress_lastdim_length = (progress < (this->former_tile) ? (this->tiled_row_length+BLOCK_DATA_NUM(float)) : this->tiled_row_length);
            if(AscendC::GetBlockIdx() == this->last_core){
                if(this->tiled_row_length == 0 && progress == (this->former_tile-1)){
                    progress_lastdim_length = this->core_tail_lastDim;
                }else if(this->tiled_row_length != 0 && progress == (this->tileNum - 1)){
                    progress_lastdim_length = progress_lastdim_length - (PAD_NUM(this->lastDim, BLOCK_BYTE, float));
                }
            }
            
            AscendC::DataCopyParams copyParams{ static_cast<uint16_t>(this->tiled_col_length), 
                                                static_cast<uint16_t>(progress_lastdim_length*sizeof(float)), 
                                                static_cast<uint16_t>((this->lastDim-progress_lastdim_length)*sizeof(float)), 
                                                0 };
            AscendC::DataCopyPadParams padParams{ true, 
                                                  0, 
                                                  static_cast<uint8_t>(PAD_NUM(progress_lastdim_length, BLOCK_BYTE, float)), 
                                                  0 };
            AscendC::DataCopyPad( xLocal, 
                                  xGm[progress < (this->former_tile) ? (progress*((this->tiled_row_length+BLOCK_DATA_NUM(float)))) : (this->former_tile*(this->tiled_row_length+BLOCK_DATA_NUM(float))+(progress-this->former_tile)*this->tiled_row_length)], 
                                  copyParams, 
                                  padParams);
            // 将Local Buffer中的数据塞到队列中，等待读取使用（EnQue的作用是将数据存到队列中）
            // AscendC::printf("tiled_col_length: %d\n", this->tiled_col_length);
            // AscendC::printf("tiled_row_num: %d\n", (progress < this->tileNum ? this->tiled_row_length : this->tail_tiled_row_length));
            // AscendC::printf("stride: %d\n", (this->lastDim-(progress < this->tileNum ? this->tiled_row_length : this->tail_tiled_row_length)));
            
            inQueueX.EnQue(xLocal); 
            break;  
        }  
        case Condition::MIDDLE: {  
            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();  
            AscendC::DataCopyParams copyParams{ static_cast<uint16_t>((progress < this->former_tile ? (this->tiled_batch+1) : this->tiled_batch)*this->tiled_col_length), 
                                                static_cast<uint16_t>(this->tiled_row_length * sizeof(float)), 
                                                0, 
                                                0 };
            AscendC::DataCopyPadParams padParams{ true, 
                                                  0, 
                                                  static_cast<uint8_t>(PAD_NUM(this->tiled_row_length, BLOCK_BYTE, float)), 
                                                  0 };
            // AscendC::printf("before_copyin\n");
            AscendC::DataCopyPad( xLocal, 
                                  xGm[this->tiled_col_length*this->tiled_row_length*(progress < (this->former_tile) ? (progress*(this->tiled_batch+1)) : (this->former_tile*(this->tiled_batch+1)+(progress-this->former_tile)*this->tiled_batch))], 
                                  copyParams, 
                                  padParams);
            // AscendC::printf("after_copyin\n");
            // 将Local Buffer中的数据塞到队列中，等待读取使用（EnQue的作用是将数据存到队列中）
            inQueueX.EnQue(xLocal); 
            break;  
        }  
        case Condition::LAST: {  
            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();  
            AscendC::DataCopyParams copyParams{ static_cast<uint16_t>(progress < this->former_tile ? (this->tiled_col_length+1) : this->tiled_col_length), 
                                                static_cast<uint16_t>(this->tiled_row_length * sizeof(float)), 
                                                0, 
                                                0 };
            AscendC::DataCopyPadParams padParams{ true, 
                                                  0, 
                                                  static_cast<uint8_t>(PAD_NUM(this->tiled_row_length, BLOCK_BYTE, float)), 
                                                  0 };
            // AscendC::printf("defore\n");
            AscendC::DataCopyPad( xLocal, 
                                  xGm[this->tiled_row_length*(progress < (this->former_tile) ? (progress*(this->tiled_col_length+1)) : ((this->former_tile)*(this->tiled_col_length+1)+(progress-this->former_tile)*this->tiled_col_length))], 
                                  copyParams, 
                                  padParams);
            // AscendC::printf("after\n");
            // 将Local Buffer中的数据塞到队列中，等待读取使用（EnQue的作用是将数据存到队列中）
            inQueueX.EnQue(xLocal); 
            break;  
        }  
        default: {  
            ASSERT(0 && "Undifined Condition Case!");  
            break;  
        }  
    }
    
  }
  __aicore__ inline void Compute(int32_t progress) {
    // 从队列中获取传入的原始数据
    AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
    // 准备本地的结果缓存，将来要用队列传出
    AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
    
    AscendC::LocalTensor<float> reduceTensor = reduceRes.Get<float>();
    // 处理softmax的核心函数
    // for(uint32_t i = 0; i < 64; i++){
    //     for(uint32_t j = 0; j < 2; j++){
    //         AscendC::printf("%f  ", xLocal(i*2+j));
    //     }
        // AscendC::printf("\n");
    // }
    DoSoftMax(xLocal, yLocal, reduceTensor, progress);
    // AscendC::printf("dosoftmax\n");

    // 将存了结果的本地buffer压入到传出队列中，等待取出写回外部存储
    outQueueY.EnQue(yLocal);
    // 计算完成后原始数据已经可以释放，free掉
    inQueueX.FreeTensor(xLocal);
  }
  __aicore__ inline void CopyOut(int32_t progress) {
    uint32_t progress_lastdim_length = 0;
    switch (this->condition) {  
        case Condition::FIRST: {  
            progress_lastdim_length = (progress < (this->former_tile) ? (this->tiled_row_length+BLOCK_DATA_NUM(float)) : this->tiled_row_length);
            if(AscendC::GetBlockIdx() == this->last_core){
                if(this->tiled_row_length == 0 && progress == (this->former_tile-1)){
                    progress_lastdim_length = this->core_tail_lastDim;
                }else if(this->tiled_row_length != 0 && progress == (this->tileNum - 1)){
                    progress_lastdim_length = progress_lastdim_length - (PAD_NUM(this->lastDim, BLOCK_BYTE, float));
                }
            }
            
            AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
            // 然后将存外的Global Memory中将本次需要处理的tile数据copy到内部的Local Buffer中
            AscendC::DataCopyParams copyParams{ static_cast<uint16_t>(this->tiled_col_length), 
                                                static_cast<uint16_t>(progress_lastdim_length * sizeof(float)), 
                                                0, 
                                                static_cast<uint16_t>((this->lastDim-progress_lastdim_length)*sizeof(float)) };
            AscendC::DataCopyPad(yGm[progress < (this->former_tile) ? (progress*((this->tiled_row_length+BLOCK_DATA_NUM(float)))) : (this->former_tile*(this->tiled_row_length+BLOCK_DATA_NUM(float))+(progress-this->former_tile)*this->tiled_row_length)], 
                                 yLocal,
                                copyParams);
            outQueueY.FreeTensor(yLocal);
            break;  
        }  
        case Condition::MIDDLE: {  
            AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
            AscendC::DataCopyParams copyParams{ static_cast<uint16_t>((progress < this->former_tile ? (this->tiled_batch+1) : this->tiled_batch)*this->tiled_col_length), 
                                                static_cast<uint16_t>(this->tiled_row_length * sizeof(float)), 
                                                0, 
                                                0 };
            AscendC::DataCopyPad(yGm[this->tiled_col_length*this->tiled_row_length*(progress < (this->former_tile) ? (progress*(this->tiled_batch+1)) : (this->former_tile*(this->tiled_batch+1)+(progress-this->former_tile)*this->tiled_batch))], 
                                 yLocal, 
                                 copyParams);
            // 将Local Buffer中的数据塞到队列中，等待读取使用（EnQue的作用是将数据存到队列中）
            outQueueY.FreeTensor(yLocal); 
            break;  
        }  
        case Condition::LAST: {  
            AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
            AscendC::DataCopyParams copyParams{ static_cast<uint16_t>(progress < this->former_tile ? (this->tiled_col_length+1) : this->tiled_col_length), 
                                                static_cast<uint16_t>(this->tiled_row_length * sizeof(float)), 
                                                0, 
                                                0 };
            
            AscendC::DataCopyPad(yGm[this->tiled_row_length*(progress < (this->former_tile) ? (progress*(this->tiled_col_length+1)) : ((this->former_tile)*(this->tiled_col_length+1)+(progress-this->former_tile)*this->tiled_col_length))], 
                                 yLocal, 
                                 copyParams);
            // 将Local Buffer中的数据塞到队列中，等待读取使用（EnQue的作用是将数据存到队列中）
            outQueueY.FreeTensor(yLocal); 
            break;  
        }  
        default: {  
            ASSERT(0 && "Undifined Condition Case!");  
            break;  
        }  
    }
  }

  template <typename T>
  __aicore__ inline void DoSoftMax(const AscendC::LocalTensor<T> &srcLocalTensor,
                                   const AscendC::LocalTensor<T> &dstLocalTensor,
                                   const AscendC::LocalTensor<T> &reduceTensor,
                                   int32_t progress) {
    uint32_t deal_length = 0;
    uint32_t src_stride = 0;
    uint32_t tile_batch_loop = 0;
    uint32_t tile_first_loop = 0;
    // AscendC::printf("block_length:%d\n", this->blockLength);
    AscendC::Exp(srcLocalTensor, srcLocalTensor, this->buffer_data_num);
    // AscendC::printf("After EXP---------\n");
    // for(uint32_t i = 0; i < 2; i++){
    //     for(uint32_t j = 0; j < 72; j++){
    //         AscendC::printf("%f  ", srcLocalTensor(i*72+j));
    //     }
    //     AscendC::printf("\n");
    // }
    // uint32_t progress_lastdim_length = 0;
    switch (this->condition)
    {
      case Condition::FIRST:
          src_stride = (progress < (this->former_tile) ? (this->tiled_row_length+BLOCK_DATA_NUM(float)) : this->tiled_row_length);
          deal_length = (progress < (this->former_tile) ? (this->tiled_row_length+BLOCK_DATA_NUM(float)) : this->tiled_row_length);
          if(AscendC::GetBlockIdx() == this->last_core){
              if(this->tiled_row_length == 0 && progress == (this->former_tile-1)){
                    deal_length = this->core_tail_lastDim;
              }else if(this->tiled_row_length != 0 && progress == (this->tileNum - 1)){
                    deal_length = deal_length - (PAD_NUM(this->lastDim, BLOCK_BYTE, float));
              }
          }
          
        //   AscendC::printf("firstDim:%d\n", this->firstDim);
        //   AscendC::printf("deal_length:%d\n", deal_length);
        //   AscendC::printf("src_stride:%d\n", src_stride);
        //   AscendC::printf("former_tile:%d\n", this->former_tile);
        //   AscendC::printf("tiled_row_length:%d\n", this->tiled_row_length);
          RollSum(srcLocalTensor, 
                  reduceTensor, 
                  this->firstDim,
                  deal_length,
                  src_stride);
          if(this->firstDim >= 64) {
              AscendC::Reciprocal(reduceTensor, reduceTensor, deal_length);
              for(uint32_t i = 0; i < this->firstDim; i++){
                  AscendC::Mul(dstLocalTensor[i*src_stride], srcLocalTensor[i*src_stride], reduceTensor, deal_length);
              }
          }else{
              for(uint32_t i = 0; i < this->firstDim; i++){
                  AscendC::Div(dstLocalTensor[i*src_stride], srcLocalTensor[i*src_stride], reduceTensor, deal_length);
              }
          }
          
        //   AscendC::printf("get one");
          
          break;
      case Condition::MIDDLE:
          tile_batch_loop = (progress < this->former_tile ? (this->tiled_batch+1) : this->tiled_batch);
        //   AscendC::printf("core_id:%d\n", AscendC::GetBlockIdx());
        //   AscendC::printf("progress:%d\n", progress);
        //   AscendC::printf("tiled_batch:%d\n", this->tiled_batch);
        //   AscendC::printf("middleDim:%d\n", this->middleDim);
        //   AscendC::printf("lastDim:%d\n", this->lastDim);
        //   AscendC::printf("block_alined_last_dim:%d\n", this->block_alined_last_dim);
        //   AscendC::printf("tile_batch_loop:%d\n", tile_batch_loop);
        //   AscendC::printf("former_tile:%d\n", this->former_tile);
          for(uint32_t i = 0; i < tile_batch_loop; i++){
              RollSum(srcLocalTensor[i*this->middleDim*this->block_alined_last_dim], 
                      reduceTensor, 
                      this->middleDim,
                      this->lastDim,
                      this->block_alined_last_dim);
            //   AscendC::printf("after_rollsum\n");
              if(this->middleDim >= 64){
                  AscendC::Reciprocal(reduceTensor, reduceTensor, this->lastDim);
                  for(uint32_t j = 0; j < this->middleDim; j++){
                      AscendC::Mul(dstLocalTensor[i*this->middleDim*this->block_alined_last_dim + j*this->block_alined_last_dim], srcLocalTensor[i*this->middleDim*this->block_alined_last_dim + j*this->block_alined_last_dim], reduceTensor, this->lastDim);
                  }
              }else{
                  for(uint32_t j = 0; j < this->middleDim; j++){
                      AscendC::Div(dstLocalTensor[i*this->middleDim*this->block_alined_last_dim + j*this->block_alined_last_dim], srcLocalTensor[i*this->middleDim*this->block_alined_last_dim + j*this->block_alined_last_dim], reduceTensor, this->lastDim);
                  }
              }
              
          }
          break;
      case Condition::LAST:
          tile_first_loop = progress < this->former_tile ? (this->tiled_col_length+1) : this->tiled_col_length;
          DoReduceSum(srcLocalTensor, reduceTensor, tile_first_loop);
          // 对每一份最后一个维度的数据求softmax
          for (uint32_t i = 0; i < tile_first_loop; i++) {
              uint32_t offset = i * this-> padded_row_length;
              AscendC::Muls(dstLocalTensor[offset], srcLocalTensor[offset],1 / reduceTensor.GetValue(i), this->lastDim);
          }
        //   AscendC::printf("lastDim:%d\n", this->lastDim);
        break;
      default:
        ASSERT(0 && "Undifined Condition Case!");
        break;
    }
  }

  template <typename T>
  __aicore__ inline void DoReduceSum(const AscendC::LocalTensor<T> &srcLocalTensor,
                                     const AscendC::LocalTensor<T> &reduceTensor,
                                     uint32_t loop_count) {
    // 对于WholeReduceSum函数而言，repeatstride表示每次向量计算之间跳过多少个block
    // blockstride表示一次向量计算中，每隔多少个block算子计算，stride=1表示连续计算
    // 如果最后一个维度大小比一次性处理的数据要多，表示一个lastdim需要多次repeat才能处理
    // 这一个if有问题 TODO:
    if (true) {
      /*
      处理多次迭代才能求和的情况，主要处理尾部数据
      */
      const uint32_t dstRepStride = 1;
      const uint32_t srcBlkStride = 1;
      // EACH_BLOCK_SIZE = 32，32Byte 一个block可以放8个float32
      // this->normalMask * sizeof(T)先算出正常处理下，一次处理的一批数据占多少个Byte
      // 然后计算这些数据一共占了多少个block
      uint32_t srcRepStride = this->normalMask * sizeof(T) / EACH_BLOCK_SIZE;
      uint32_t local_loop = this->lastDim / this->normalMask;
      uint32_t tail_mask = this->lastDim % this->normalMask;
      for(uint32_t i = 0; i < loop_count; i++){
          T Sum = 0;
          for(uint32_t j = 0; j < local_loop; j++){
              AscendC::WholeReduceSum(reduceTensor[i], srcLocalTensor[i*this->padded_row_length + j*this->normalMask], this->normalMask, 1, dstRepStride, srcBlkStride, srcRepStride);
              Sum += reduceTensor[i](0);
          }
          if(tail_mask != 0){
              // 剩余数据不足一个数据block，需要标量处理
              if(tail_mask < (EACH_BLOCK_SIZE / 4)){
                  for(uint32_t j = 0; j < tail_mask; j++){
                      Sum += srcLocalTensor.GetValue(i*this->padded_row_length + local_loop*this->normalMask + j);
                      if(srcLocalTensor.GetValue(i*this->padded_row_length + local_loop*this->normalMask + j) == 1.f){
                            // AscendC::printf("find location: %d\n", local_loop*this->normalMask + j);
                            // AscendC::printf("padded_row_length: %d\n", this->padded_row_length);
                            // AscendC::printf("tail_mask: %d\n", tail_mask);
                            // AscendC::printf("local_loop: %d\n", local_loop);
                            // AscendC::printf("normalMask: %d\n", this->normalMask);
                      }
                  }
              }
              // 否则向量处理
              else{
                  AscendC::WholeReduceSum(reduceTensor[i], srcLocalTensor[i*this->padded_row_length + local_loop*this->normalMask], tail_mask, 1, dstRepStride, srcBlkStride, srcRepStride);
                  Sum += reduceTensor[i](0);
              }
          }
          reduceTensor[i](0) = Sum;
        //   AscendC::printf("Sum:%f\n", Sum);
      }
      
      
      // AscendC::WholeReduceSum(reduceTensor, srcLocalTensor, this->normalMask,
      //                repeatTimes, dstRepStride, srcBlkStride, srcRepStride);
    }
  }
  template <typename T>
  __aicore__ inline void RollSum(const AscendC::LocalTensor<T> &srcLocalTensor,
                                 const AscendC::LocalTensor<T> &reduceTensor,
                                 uint32_t loop_count,
                                 uint32_t deal_row_length,
                                 uint32_t stride_length){
      T scalar = 0;
    //   AscendC::printf("before_duplicate\n");
      AscendC::Duplicate(reduceTensor, scalar, stride_length);
    //   AscendC::printf("after_duplicate\n");
      for(uint32_t i = 0; i < loop_count; i++){
          AscendC::Add(reduceTensor, reduceTensor, srcLocalTensor[i*stride_length], deal_row_length);
        //   AscendC::printf("after_Add:%d\n", i);
      }
  }


 private:
  uint32_t totalLength;
  uint32_t dimNum;
  uint32_t firstDim;
  uint32_t middleDim;
  uint32_t lastDim;
  uint32_t dim;
  uint32_t tileNum;
  uint32_t condition;
  uint32_t blockLength;
  uint32_t tileLength;

  uint32_t outputLength;
  uint32_t outputTileLength;

  uint32_t k;
  uint32_t blockReduceStride;
  uint32_t normalMask;
  uint16_t RepeatTimes;
  uint32_t padTilingLength;

  uint32_t padded_row_length;
  // uint32_t tile_rows;
  uint32_t buffer_data_num;

  uint32_t tiled_batch;
  uint32_t tiled_row_length;
  uint32_t tiled_col_length;
  uint32_t tail_tiled_batch;
  uint32_t tail_tiled_row_length;
  uint32_t tail_tiled_col_length;
  uint32_t block_alined_last_dim;
  bool have_tail;

  uint32_t former_tile;
  uint32_t loop_count;

  uint32_t AI_core_num;
  uint32_t core_last_dim_block_num;
  uint32_t former_core_num;
  uint32_t core_firstDim;
  uint32_t core_middleDim;
  uint32_t core_lastDim;
  uint32_t core_tail_lastDim;
  uint32_t last_core;

  AscendC::GlobalTensor<float> xGm;
  AscendC::GlobalTensor<float> yGm;


  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueY;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> reduceRes;
};
extern "C" __global__ __aicore__ void soft_max(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    // GET_TILING_DATA(tiling_data, tiling);
    KernelSoftmax op;
    op.Init(x, y, 
            tiling_data.totalLength, 
            tiling_data.dimNum,
            tiling_data.firstDim,
            tiling_data.middleDim,
            tiling_data.lastDim,
            tiling_data.dim,
            tiling_data.tileNum,
            tiling_data.condition,
            tiling_data.dtype);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}