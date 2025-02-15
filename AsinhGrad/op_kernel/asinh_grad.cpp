// #include "stdio.h"
#include "kernel_operator.h"
// #include "tiling/tiling_api.h"


constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue

class KernelAsinhGrad {
public:
    __aicore__ inline KernelAsinhGrad() {}
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR y, GM_ADDR z,
                                uint32_t totalLength, uint32_t ALIGN_NUM,
                                uint32_t block_size, uint32_t core_size,
                                uint32_t core_remain) {
        this->blockLength = core_size + (AscendC::GetBlockNum() == AscendC::GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * AscendC::GetBlockIdx();
        auto bufferlength = this->blockLength;

        // get start index for current core, core parallel
        dyGm.SetGlobalBuffer((__gm__ DTYPE_DY*)dy + startPointer, bufferlength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + startPointer, bufferlength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z*)z + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueDY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_DY));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
        pipe.InitBuffer(tmpBuffer, this->tileLength * sizeof(DTYPE_DY));
        pipe.InitBuffer(tmpBufferfp32_1, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBufferfp32_2, this->tileLength * sizeof(float));
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
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length)
    {
        AscendC::LocalTensor<DTYPE_DY> dyLocal = inQueueDY.AllocTensor<DTYPE_DY>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();

        AscendC::DataCopy(dyLocal, dyGm[progress * this->tileLength], length);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], length);

        inQueueDY.EnQue(dyLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length)
    {
        AscendC::LocalTensor<DTYPE_DY> dyLocal = inQueueDY.DeQue<DTYPE_DY>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::LocalTensor<DTYPE_Y> tmp = tmpBuffer.Get<DTYPE_DY>();

        AscendC::LocalTensor<float> tmpfp32_1 = tmpBufferfp32_1.Get<float>();
        AscendC::LocalTensor<float> tmpfp32_2 = tmpBufferfp32_2.Get<float>();

        // if(sizeof(DTYPE_Y) == 2){

        //     DTYPE_Y c2 = -1.00;
        //     float c3 = 0.50;

        //     // AscendC::Cast(dstLocal, srcLocal, AscendC::RoundMode::CAST_CEIL, 512);
        //     // AscendC::DumpTensor(yLocal,1,length);
        //     AscendC::Cast(tmpfp32_1, yLocal, AscendC::RoundMode::CAST_NONE ,length);
        //     // AscendC::DumpTensor(tmpfp32_1,1,length);
        //     // AscendC::DataCopy(tmp,yLocal, length);
        //     // AscendC::DumpTensor(tmp,1,length);
        //     // AscendC::DumpTensor(yLocal,1,length);
        //     AscendC::Exp(tmpfp32_1,tmpfp32_1,length); //tmp = e^y
        //     // AscendC::Cast(tmp, tmpfp32_1, AscendC::RoundMode::CAST_ODD, length);
        //     // AscendC::Muls(tmp, tmp, c3, length);
        //     // AscendC::DumpTensor(yLocal,1,length);

        //     AscendC::Muls(yLocal, yLocal, c2, length); // y = - y
        //     AscendC::Cast(tmpfp32_2, yLocal, AscendC::RoundMode::CAST_NONE, length);
        //     // AscendC::DumpTensor(yLocal,1,length);

        //     AscendC::Exp(tmpfp32_2,tmpfp32_2,length);  //y = e^-y
        //     // AscendC::Cast(yLocal, tmpfp32_1, AscendC::RoundMode::CAST_ODD, length);
        //     // AscendC::Muls(yLocal, yLocal, c3, length);
        //     // AscendC::DumpTensor(yLocal,1,length);

        //     AscendC::Add(tmpfp32_1,tmpfp32_1,tmpfp32_2,length);   //(e^y+e^-y)
        //     // AscendC::DumpTensor(yLocal,1,length);

        //     AscendC::Muls(tmpfp32_1, tmpfp32_1, c3, length); //(e^y+e^-y)/2
        //     // AscendC::DumpTensor(tmp,1,length);

        //     AscendC::Cast(tmpfp32_2, dyLocal, AscendC::RoundMode::CAST_NONE, length);

        //     AscendC::Div(tmpfp32_1,tmpfp32_2,tmpfp32_1,length); //dy/((e^y+e^-y)/2) 
        //     AscendC::Cast(zLocal, tmpfp32_1, AscendC::RoundMode::CAST_NONE, length);
        if constexpr (std::is_same_v<DTYPE_Y, half>){
            float c3 = 0.50, c2 = -1.00;
            AscendC::Cast(tmpfp32_1, yLocal, AscendC::RoundMode::CAST_NONE ,length);
            AscendC::DataCopy(tmpfp32_2,tmpfp32_1, length);
            AscendC::Exp(tmpfp32_1,tmpfp32_1,length); //tmpfp32_1 = e^y
            AscendC::Muls(tmpfp32_2, tmpfp32_2, c2, length);
            AscendC::Exp(tmpfp32_2,tmpfp32_2,length);  //y = e^-y
            AscendC::Add(tmpfp32_1,tmpfp32_1,tmpfp32_2,length);   //(e^y+e^-y)
            AscendC::Muls(tmpfp32_1, tmpfp32_1, c3, length); //(e^y+e^-y)/2
            AscendC::Cast(tmpfp32_2, dyLocal, AscendC::RoundMode::CAST_NONE, length);
            AscendC::Div(tmpfp32_1,tmpfp32_2,tmpfp32_1,length);
            AscendC::Cast(zLocal, tmpfp32_1, AscendC::RoundMode::CAST_NONE, length);
        }else if constexpr (std::is_same_v<DTYPE_Y, float>){
            DTYPE_Y c2 = -1.00, c3 = 0.50;
            AscendC::Exp(tmp,yLocal,length);
            AscendC::Muls(yLocal, yLocal, c2, length);
            AscendC::Exp(yLocal,yLocal,length); 
            AscendC::Add(yLocal,yLocal,tmp,length);
            AscendC::Muls(yLocal, yLocal, c3, length);
            AscendC::Div(zLocal,dyLocal,yLocal,length);
        }
        // }
        // AscendC::DumpTensor(zLocal,1,length);
        // AscendC::printf("output: %f \n",zLocal[1] );

        // AscendC::Muls(zLocal,tmp,c4,length);

        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueDY.FreeTensor(dyLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, length);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffer, tmpBufferfp32_1, tmpBufferfp32_2; //signbitBuffer
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueDY, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    AscendC::GlobalTensor<DTYPE_DY> dyGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    // AscendC::LocalTensor<DTYPE_Y> signbit;
};

extern "C" __global__ __aicore__ void asinh_grad(GM_ADDR dy, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
        KernelAsinhGrad op;
    
    op.Init(y,dy, z, tiling_data.totalLength, 
            tiling_data.ALIGN_NUM, tiling_data.block_size,
            tiling_data.core_size, tiling_data.core_remain);
    op.Process();
}