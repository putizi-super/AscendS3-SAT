#include <type_traits>
#include "kernel_operator.h"

template<typename T>
class ReplicationPad2dKernel {
public:
    __aicore__ inline ReplicationPad2dKernel() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, 
            int32_t padL, int32_t padR, int32_t padT, int32_t padB, 
            int32_t Xshape[], int32_t Yshape[], uint32_t blocksize, 
            int32_t XDim, int32_t YDim, 
            uint32_t blockLengthMean, uint32_t blockLengthEnd,
            int32_t totalSizeX, int32_t totalSizeY, 
            uint32_t lastDim, int32_t lastDimY) {

        this->padL = padL;
        this->padR = padR;
        this->padT = padT;
        this->padB = padB;
        this->lastDim = lastDim;
        this->lastDimY = lastDimY;

        for(int32_t i = 0; i < 10; i++) {
            ((int32_t *)this->Xshape)[i] = Xshape[i];
            ((int32_t *)this->Yshape)[i] = Yshape[i];
        }
        this->blocksize = blocksize;
        this->XDim = XDim;
        this->YDim = YDim;
        this->lastNum = 1;
        for(int32_t i = 0; i < XDim - 1; i++){
            this->lastNum *= Xshape[i];
            this->lastNumY *= Yshape[i];
        }
        this->blockLengthMean = blockLengthMean;
        this->blockLengthEnd = blockLengthEnd;
        this->totalSizeX = totalSizeX;
        this->totalSizeY = totalSizeY;

        this->Gm_X.SetGlobalBuffer((__gm__ T*)x, this->totalSizeX);
        this->Gm_Y.SetGlobalBuffer((__gm__ T*)y, this->totalSizeY);
        
        // 1. 关于 queBind 以及 非连续的 copydata
        // this->pipe.InitBuffer(tmpBuf1, this->blockLengthMean * sizeof(T));
        // this->pipe.InitBuffer(tmpBuf2, this->blockLengthEnd * sizeof(T));

        // this->pipe.InitBuffer(inQueueSrc, 1, this->blocksize * sizeof(T) * 1024);
        this->pipe.InitBuffer(queBind, 1, this->blocksize * sizeof(T) * this->lastDimY);
        // this->pipe.InitBuffer(outQueueDst, 1, this->blocksize * sizeof(T) * 1024);
    }

    __aicore__ inline void Process() {
        if(this->XDim == 4){
            this->Process4D();
        }else{
            if(this->XDim == 3){
                this->Process3D();
            }
        }
    }
private:
    __aicore__ inline void Process4D() {
        // Calculate the dimensions of the input and output tensors
        int32_t batchSize = Xshape[0];     // Batch size is the first dimension
        int32_t channels = Xshape[1];      // Channels is the second dimension
        int32_t inputHeight = Xshape[2];   // Height is the third dimension
        int32_t inputWidth = Xshape[3];    // Width is the fourth dimension
        int32_t outputHeight = Yshape[2];  // Height is the third dimension
        int32_t outputWidth = Yshape[3];   // Width is the fourth dimension

        // 2. 关于数据同步, 不应该从 GM_Y 中取数据, 应该都从 GM_X 中取数据, 使得数据之间不会产生冲突
        // 3. 数据拷贝还是稍微慢一些，相对于一些循环。如果 先左右填充，再上下填充，然后从GM_Y 中取数据的话, 会取到很多 零, 导致结果错误
        // 4. 如果从GM_Y 中取数据, 就需要考虑如何避免数据冲突, 使用         // AscendC::TQueSync<PIPE_S, PIPE_S> sync; // 肯定会造成性能损失
        // original data
        uint32_t dstStride = padR + padL; // 下一行的起始数据
        uint16_t blockCount = inputHeight;
        for(int i = 0; i < batchSize * channels; ++i){
            int32_t inIndex = i * (inputHeight * inputWidth);
            int32_t outIndex = i * (outputHeight * outputWidth) + padT * outputWidth + padL; 
            // AscendC::PRINTF("inIndex: %d, outIndex: %d\n", inIndex, outIndex);           
            CopyIn(inIndex, blockCount, 0);
            CopyOut(outIndex, blockCount, dstStride);
        }

        // Top padding
        blockCount = channels * batchSize;
        uint32_t Stride = outputWidth * (outputHeight - 1);
        for(int32_t i = 0; i < padT; i++){
            int32_t inIndex = padT * outputWidth;
            int32_t outIndex = i * outputWidth;
            CopyInTopBottom(inIndex, blockCount, Stride);
            CopyOutTopBottom(outIndex, blockCount, Stride);           
        }
        // Bottom padding
        for(int32_t i = 0; i < padB; i++){
            int32_t inIndex = (padT + inputHeight - 1) * outputWidth;
            int32_t outIndex = (padT + inputHeight + i) * outputWidth;
            CopyInTopBottom(inIndex, blockCount, Stride);
            CopyOutTopBottom(outIndex, blockCount, Stride);
        }

        // pading right and left
        for (int32_t b = 0; b < batchSize; ++b) {
            for (int32_t c = 0; c < channels; ++c) {
                for (int32_t i = 0; i < outputHeight; ++i) {
                    int32_t srcIndex;
                    // Calculate srcIndex for left and right padding
                    if (i < padT) {
                        // Top padding: use the first row of input
                        srcIndex = b * channels * inputHeight * inputWidth + c * inputHeight * inputWidth;
                    } else if (i >= padT + inputHeight) {
                        // Bottom padding: use the last row of input
                        srcIndex = b * channels * inputHeight * inputWidth + c * inputHeight * inputWidth + (inputHeight - 1) * inputWidth;
                    } else {
                        // Middle: use the corresponding row of input
                        srcIndex = b * channels * inputHeight * inputWidth + c * inputHeight * inputWidth + (i - padT) * inputWidth;
                    }

                    // Fill the left padding
                    T value = this->Gm_X.GetValue(srcIndex);
                    for (int32_t j = 0; j < padL; ++j) {
                        int32_t dstIndex = b * channels * outputHeight * outputWidth + c * outputHeight * outputWidth + i * outputWidth + j;
                        this->Gm_Y.SetValue(dstIndex, value);
                    }

                    // Fill the right padding
                    if (i < padT) {
                        // Top padding: use the first row, last column of input
                        srcIndex = b * channels * inputHeight * inputWidth + c * inputHeight * inputWidth + (inputWidth - 1);
                    } else if (i >= padT + inputHeight) {
                        // Bottom padding: use the last row, last column of input
                        srcIndex = b * channels * inputHeight * inputWidth + c * inputHeight * inputWidth + (inputHeight - 1) * inputWidth + (inputWidth - 1);
                    } else {
                        // Middle: use the corresponding row, last column of input
                        srcIndex = b * channels * inputHeight * inputWidth + c * inputHeight * inputWidth + (i - padT) * inputWidth + (inputWidth - 1);
                    }

                    value = this->Gm_X.GetValue(srcIndex);
                    for (int32_t j = outputWidth - padR; j < outputWidth; ++j) {
                        int32_t dstIndex = b * channels * outputHeight * outputWidth + c * outputHeight * outputWidth + i * outputWidth + j;
                        this->Gm_Y.SetValue(dstIndex, value);
                    }
                }
            }
        }

        // AscendC::TQueSync<PIPE_S, PIPE_S> sync;
        // sync.SetFlag(0);
        // sync.WaitFlag(0);
    }

    __aicore__ inline void Process3D() {
        // Calculate the dimensions of the input and output tensors
        int32_t channels = Xshape[0];      // Channels is the first dimension
        int32_t inputHeight = Xshape[1];   // Height is the second dimension
        int32_t inputWidth = Xshape[2];    // Width is the third dimension
        int32_t outputHeight = Yshape[1];  // Height is the second dimension
        int32_t outputWidth = Yshape[2];   // Width is the third dimension

        // original data
        uint32_t dstStride = padR + padL; // 下一行的起始数据
        uint16_t blockCount = inputHeight;
        for(int i = 0; i < channels; ++i){
            int32_t inIndex = i * (inputHeight * inputWidth);
            int32_t outIndex = i * (outputHeight * outputWidth) + padT * outputWidth + padL; 
            // AscendC::PRINTF("inIndex: %d, outIndex: %d\n", inIndex, outIndex);           
            CopyIn(inIndex, blockCount, 0);
            CopyOut(outIndex, blockCount, dstStride);
        }

        // Top padding
        blockCount = channels;
        uint32_t Stride = outputWidth * (outputHeight - 1);
        for(int32_t i = 0; i < padT; i++){
            int32_t inIndex = padT * outputWidth;
            int32_t outIndex = i * outputWidth;
            CopyInTopBottom(inIndex, blockCount, Stride);
            CopyOutTopBottom(outIndex, blockCount, Stride);           
        }
        // Bottom padding
        for(int32_t i = 0; i < padB; i++){
            int32_t inIndex = (padT + inputHeight - 1) * outputWidth;
            int32_t outIndex = (padT + inputHeight + i) * outputWidth;
            CopyInTopBottom(inIndex, blockCount, Stride);
            CopyOutTopBottom(outIndex, blockCount, Stride);
        }

        // pading right and left
        for (int32_t c = 0; c < channels; ++c) {
            for (int32_t i = 0; i < outputHeight; ++i) {
                int32_t srcIndex;
                // Calculate srcIndex for left and right padding
                if (i < padT) {
                    // Top padding: use the first row of input
                    srcIndex = c * inputHeight * inputWidth;
                } else if (i >= padT + inputHeight) {
                    // Bottom padding: use the last row of input
                    srcIndex = c * inputHeight * inputWidth + (inputHeight - 1) * inputWidth;
                } else {
                    // Middle: use the corresponding row of input
                    srcIndex = c * inputHeight * inputWidth + (i - padT) * inputWidth;
                }

                // Fill the left padding
                T value = this->Gm_X.GetValue(srcIndex);
                for (int32_t j = 0; j < padL; ++j) {
                    int32_t dstIndex = c * outputHeight * outputWidth + i * outputWidth + j;
                    this->Gm_Y.SetValue(dstIndex, value);
                }

                // Fill the right padding
                if (i < padT) {
                    // Top padding: use the first row, last column of input
                    srcIndex = c * inputHeight * inputWidth + (inputWidth - 1);
                } else if (i >= padT + inputHeight) {
                    // Bottom padding: use the last row, last column of input
                    srcIndex = c * inputHeight * inputWidth + (inputHeight - 1) * inputWidth + (inputWidth - 1);
                } else {
                    // Middle: use the corresponding row, last column of input
                    srcIndex = c * inputHeight * inputWidth + (i - padT) * inputWidth + (inputWidth - 1);
                }

                value = this->Gm_X.GetValue(srcIndex);
                for (int32_t j = outputWidth - padR; j < outputWidth; ++j) {
                    int32_t dstIndex = c * outputHeight * outputWidth + i * outputWidth + j;
                    this->Gm_Y.SetValue(dstIndex, value);
                }
            }
        }
    } 
    __aicore__ inline void CopyIn(int32_t index, uint16_t blockCount, uint32_t srcStride) {
        AscendC::LocalTensor<T> srcLocal = queBind.AllocTensor<T>(); 
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(blockCount),  
            static_cast<uint32_t>(this->lastDim * sizeof(T)), 
            static_cast<uint32_t>(srcStride * sizeof(T)), 
            0, 
            0
        }; // 结构体DataCopyExtParams最后一个参数是rsv保留位
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(srcLocal, this->Gm_X[index], copyParams, padParams); // 从GM->VECIN搬运40Bytes
        queBind.EnQue<T>(srcLocal);  
    }

    __aicore__ inline void CopyOut(int32_t index, uint16_t blockCount, uint32_t dstStride) {
        AscendC::LocalTensor<T> dstLocal = queBind.DeQue<T>();
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(blockCount),  
            static_cast<uint32_t>(this->lastDim * sizeof(T)), 
            0, 
            static_cast<uint32_t>(dstStride * sizeof(T)), 
            0
        }; 
        AscendC::DataCopyPad(this->Gm_Y[index], dstLocal, copyParams); // 从GM->VECIN搬运40Bytes
        queBind.FreeTensor(dstLocal);
    }

    __aicore__ inline void CopyInTopBottom(int32_t index, uint16_t blockCount,uint32_t srcStride) {
        AscendC::LocalTensor<T> srcLocal = queBind.AllocTensor<T>(); 
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(blockCount), 
            static_cast<uint32_t>(this->lastDimY * sizeof(T)), 
            static_cast<uint32_t>(srcStride * sizeof(T)), 
            0, 
            0
        }; 
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(srcLocal, this->Gm_Y[index], copyParams, padParams); // 从GM->VECIN搬运40Bytes
        queBind.EnQue<T>(srcLocal);  
    }

    __aicore__ inline void CopyOutTopBottom(int32_t index, uint16_t blockCount,uint32_t dstStride) {
        AscendC::LocalTensor<T> dstLocal = queBind.DeQue<T>();
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(blockCount), 
            static_cast<uint32_t>(this->lastDimY * sizeof(T)), 
            0, 
            static_cast<uint32_t>(dstStride * sizeof(T)), 
            0
        }; 
        AscendC::DataCopyPad(this->Gm_Y[index], dstLocal, copyParams); 
        queBind.FreeTensor(dstLocal);
    }
private:
    AscendC::GlobalTensor<T> Gm_X;
    AscendC::GlobalTensor<T> Gm_Y;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 1> queBind; // 使用TQueBind替换原来QueI，QueO
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueSrc;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueDst;
    AscendC::DataCopyPadExtParams<T> padParams;
    AscendC::DataCopyExtParams copyParams;
    T scalar = 0;
    int32_t Xshape[10];
    int32_t Yshape[10];
    int32_t padL;
    int32_t padR;
    int32_t padT;
    int32_t padB;

    uint32_t blocksize;
    int32_t XDim;
    int32_t YDim;
    uint32_t blockLengthMean;
    uint32_t blockLengthEnd;

    uint32_t lastDim;
    uint32_t lastDimY;
    uint32_t lastNum;
    uint32_t lastNumY;
    int32_t totalSizeX;
    int32_t totalSizeY;
};
extern "C" __global__ __aicore__ void replication_pad2d(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    ReplicationPad2dKernel<DTYPE_X> op;
    op.Init(x, paddings, y, tiling_data.padL, tiling_data.padR, tiling_data.padT, tiling_data.padB, 
        tiling_data.Xshape, tiling_data.Yshape, tiling_data.blocksize, tiling_data.XDim, tiling_data.YDim, 
        tiling_data.blockLengthMean, tiling_data.blockLengthEnd, tiling_data.totalSizeX, tiling_data.totalSizeY, 
        tiling_data.lastDim, tiling_data.lastDimY);
    op.Process();

}