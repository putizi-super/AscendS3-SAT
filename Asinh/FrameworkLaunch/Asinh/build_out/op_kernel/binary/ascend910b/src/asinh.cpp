#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue

template<typename TYPE_X, typename TYPE_Y>
class KernelAsinh {

public:
    __aicore__ inline KernelAsinh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, 
                                uint32_t ALIGN_NUM, uint32_t block_size, 
                                uint32_t core_size, uint32_t core_remain) {
        this->blockLength = core_size + (AscendC::GetBlockNum() == AscendC::GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * AscendC::GetBlockIdx();
        auto bufferlength = this->blockLength;

        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + startPointer, bufferlength);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
        pipe.InitBuffer(tmpBuffer, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer_2, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer_3, this->tileLength * sizeof(float));
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount-1; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        auto length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length);
        CopyOut(loopCount - 1, length);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], length);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();
        AscendC::LocalTensor<float> tmp = tmpBuffer.Get<float>();
        AscendC::LocalTensor<float> tmp_2 = tmpBuffer_2.Get<float>();
        AscendC::LocalTensor<float> tmp_3 = tmpBuffer_3.Get<float>();

        float c1 = 1.0;
        
        if constexpr (std::is_same_v<TYPE_X, half>){
            AscendC::Cast(tmp_3, xLocal, AscendC::RoundMode::CAST_NONE, length);
            AscendC::Abs(tmp, tmp_3, length);
            AscendC::Div(tmp_2, tmp_3, tmp, length);
            // AscendC::Sign(tmp_2, xLocal, tmp_3);
            AscendC::Mul(tmp_3, tmp_3, tmp_3, length);
            AscendC::Adds(tmp_3, tmp_3, c1, length);
            AscendC::Sqrt(tmp_3, tmp_3, length);
            AscendC::Add(tmp_3, tmp_3, tmp, length);
            AscendC::Ln(tmp_3, tmp_3, length);
            AscendC::Mul(tmp_3, tmp_3, tmp_2, length);
            AscendC::Cast(yLocal, tmp_3, AscendC::RoundMode::CAST_NONE, length);
        } else if constexpr (std::is_same_v<TYPE_X, float>){
            AscendC::Abs(tmp, xLocal, length);
            AscendC::Div(tmp_2, xLocal, tmp, length);
            // AscendC::Sign(tmp_2, xLocal, tmp_3);
            AscendC::Mul(xLocal, xLocal, xLocal, length);
            AscendC::Adds(xLocal, xLocal, c1, length);
            AscendC::Sqrt(xLocal, xLocal, length);
            AscendC::Add(xLocal, xLocal, tmp, length);
            AscendC::Ln(xLocal, xLocal, length);
            AscendC::Mul(yLocal, xLocal, tmp_2, length);
        }
        
    
        outQueueY.EnQue<TYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, length);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffer, tmpBuffer_2, tmpBuffer_3;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void asinh(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelAsinh<DTYPE_X, DTYPE_Y> op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, 
            tiling_data.block_size, tiling_data.core_size, 
            tiling_data.core_remain);
    op.Process();
}

