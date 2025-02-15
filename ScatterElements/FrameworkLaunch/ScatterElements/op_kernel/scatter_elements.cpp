#include "kernel_operator.h"
#include <cstdlib>
#include <type_traits>
constexpr uint32_t BLOCK_SIZE = 32;
constexpr int32_t BUFFER_NUM = 2;  // tensor num for each queue
template<typename T> struct map {
    using type = float;
};
template<> struct map<int32_t> {
    using type = int64_t;
};
template<> struct map<uint8_t> {
    using type = uint32_t;
};

template<typename TYPE_VAR, typename TYPE_INDICES, typename TYPE_UPDATES> class KernelScatterElements {
    using T = TYPE_VAR;
public:
    __aicore__ inline KernelScatterElements() {}

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, int32_t axis, int32_t mode,int32_t ss[], 
                                int32_t size[], int32_t ndims[],
                                uint32_t total_length, uint32_t tile_num_mean,
                                uint32_t tile_num_end, uint32_t tile_length_mean,
                                uint32_t tile_length_end) {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        ResovleTiling(total_length, tile_num_mean, tile_num_end, tile_length_mean,tile_length_end);
        for(int32_t i=0; i<3; ++i){
            ((int32_t *)this->totalSize)[i] = size[i];
            ((int32_t *)this->totaldims)[i] = ndims[i];
            // AscendC::PRINTF("%d %d\n", size[i], ndims[i]);
        }
        // if constexpr (std::is_same_v<T, half>) {
        //     pipe.InitBuffer(x1_inque, BUFFER_NUM, 9999999999 * sizeof(T));
        //     pipe.InitBuffer(x2_inque, BUFFER_NUM, 9999999999 * sizeof(T));
        //     pipe.InitBuffer(x1_inque, BUFFER_NUM, 9999999999 * sizeof(T));

        // }
        this->var_length = size[0];
        this->indices_length = size[1];
        this->updates_length = size[2];
        // AscendC::PRINTF("GetBlockNum %d\n", AscendC::GetBlockNum());
        // AscendC::PRINTF("this->block_length %d\n", this->block_length);
        Gm_var.SetGlobalBuffer((__gm__ TYPE_VAR*)var , totalSize[0]);
        Gm_indices.SetGlobalBuffer((__gm__ TYPE_INDICES*)indices + this->block_length * AscendC::GetBlockIdx(), this->block_length);
        Gm_updates.SetGlobalBuffer((__gm__ TYPE_UPDATES*)updates , totalSize[2]);
        this->axis = axis;
        // 该算子有两种策略: NONE 模式（这里默认为 assign 模式）
        // 1. NONE 模式（mode = 1）：直接将 updates 写入到 var 中，即 assign 模式
        // 2. add 模式（mode = 2）：将 updates 加到 var 中
        // 3. multiply 模式（mode = 3）：将 updates 乘到 var 中
        this->mode = mode;
        for (int32_t i = 0; i < 192; ++i){
            ((int32_t *)this->shape)[i] = ss[i];
        }
        // 初始化 var_idex 以及 updates_index
        for(int32_t i=0; i<64; ++i){
            var_index[i] = 0;
            updates_index[i] = 0;
        }

        // for(int32_t i = 0; i < 32; ++i){
        //     auto a1 = Gm_updates.GetValue(i);
        //     auto a2 = Gm_var.GetValue(i); 
        //     auto a3 = Gm_indices.GetValue(i);
        //     AscendC::PRINTF("update %f\n", a1);
        //     AscendC::PRINTF("var %f\n", a2);
        //     AscendC::PRINTF("indices %d\n", a3);                       
        // }

        // this->blockLength = core_size + (AscendC::GetBlockNum() == AscendC::GetBlockIdx() + 1 ? core_remain : 0);
        // this->tileLength = block_size;
        // this->lastdim = lastdim;
        // this->totalLength = totalLength;
        // this->ALIGN_NUM = ALIGN_NUM;
        // if(this->mode == 1){
        //     this->tileNum = this->lastdim / this->tileLength + (this->lastdim % this->tileLength > 0);
        // }else{
        //     this->tileNum = this->lastdim / this->tileLength + (this->lastdim % this->tileLength > 0);
        // }
    }
    __aicore__ inline void Process() {
        using F = typename map<T>::type;
        //  要求： totaldims[0] == totaldims[1] == totaldims[2] 不支持广播  
        // AscendC::PRINTF("totaldims[0] %d\n", totaldims[1]);
        // AscendC::PRINTF("totaldims[1] %d\n", totaldims[1]);
        // AscendC::PRINTF("totaldims[2] %d\n", totaldims[2]);
        // AscendC::PRINTF("totalSize[0] %d\n", totalSize[0]);
        // AscendC::PRINTF("totalSize[1] %d\n", totalSize[1]);
        // AscendC::PRINTF("totalSize[2] %d\n", totalSize[2]);

        // totaldims[0] == totaldims[1] == totaldims[2]
        for(int32_t flat_index = 0; flat_index < totalSize[1]; ++flat_index){

            for(int32_t k = 0; k < totaldims[0] ; ++k){
                var_index[k] = updates_index[k];
            } 

            // AscendC::PRINTF("flat_index %d\n", flat_index);

            // update var index
            TYPE_INDICES new_idx = Gm_indices.GetValue(flat_index);
            var_index[axis] = new_idx;

            // 打印 var_index 和 updates_index`
            // AscendC::PRINTF("var_index :");
            // for(int32_t i=0; i<totaldims[0]; ++i){
            //     AscendC::PRINTF("%d ", var_index[i]);
            // }
            // // AscendC::PRINTF("\n");
            // AscendC::PRINTF("updates_index :");
            // for(int32_t i=0; i<totaldims[2]; ++i){
            //     AscendC::PRINTF("%d ", updates_index[i]);
            // }
            // AscendC::PRINTF("\n");

            // get var flat_index
            int32_t flat_index_var = 0;
            int32_t stride = 1;
            for(int32_t k = totaldims[0] -1; k >= 0; --k){
                flat_index_var += var_index[k] * stride;
                stride *= shape[0][k];
            } 
            // get updates flat_index
            int32_t flat_index_updates = 0;
            stride = 1;
            for(int32_t k = totaldims[2] -1; k >= 0; --k){
                flat_index_updates += updates_index[k] * stride;
                stride *= shape[2][k];
            }

            // AscendC::PRINTF("flat_index_updates %d, flat_index_var %d\n", flat_index_updates, flat_index_var);
            // update var value according to mode           
            F a1 = Gm_updates.GetValue(flat_index_updates);
            F a2 = Gm_var.GetValue(flat_index_var);
            
            // AscendC::PRINTF("a1 %f\n", a1);
            // AscendC::PRINTF("a2 %f\n", a2);
            if(this->mode == 2){
                auto result = a1 + a2;
                // AscendC::PRINTF("result %f\n", result);
                Gm_var.SetValue(flat_index_var, (T)result);
            }else if(this->mode == 3){
                auto result = a1 * a2;
                // AscendC::PRINTF("result %f\n", result);
                Gm_var.SetValue(flat_index_var, (T)result);
            }else{
                auto result = a1;
                // AscendC::PRINTF("result %f\n", result);
                Gm_var.SetValue(flat_index_var, (T)result);
            }
            // 坐标 + 1 并且调整
            updates_index[totaldims[1] - 1]++;  // 最后一维度加1
            // var_index[totaldims[1] - 1]++; 
            for(int32_t j = totaldims[1] - 1; j > 0; --j){
                if(updates_index[j] >= shape[1][j]){
                    updates_index[j] = 0;
                    updates_index[j - 1]++;  //  carry
                    // var_index[j] = 0;
                    // var_index[j - 1]++;
                }
            } 
        }
    }
    __aicore__ inline void Process_muti_core() {
        using F = typename map<T>::type;

        int32_t flat_index_begin = this->block_id * this->block_length;

        // AscendC::PRINTF("flat_index_begin %d\n", flat_index_begin);
        // AscendC::PRINTF("tile_length %d\n", this->tile_length);
        // 对 tile 内部遍历
        if(this->updates_length != 1){
            for(int32_t flat_index = 0; flat_index < this->tile_length ; ++flat_index){
                // AscendC::PRINTF("flat_index %d\n", flat_index);
                int32_t flat_index_true = flat_index_begin + flat_index;
                
                // get updates_index according to flat_index_begin: 这里根据 indices 的shape 获得 updates 与 indices的共同索引
                int32_t stride = 1;
                for(int32_t k = totaldims[1] - 1; k >= 0; --k){
                    updates_index[k] = (flat_index_true / stride) % shape[1][k];
                    var_index[k] = updates_index[k];
                    stride *= shape[1][k];
                }
                // update var_index according to updates_index
                // for(int32_t k = 0; k < totaldims[0] ; ++k){
                //     var_index[k] = updates_index[k];
                // } 
                // AscendC::PRINTF("flat_index %d\n", flat_index);
                // AscendC::printf("flat_index %d\n", flat_index);

                // update var index
                TYPE_INDICES new_idx = Gm_indices.GetValue(flat_index);
                var_index[axis] = new_idx;

                // 打印 var_index 和 updates_index`
                // AscendC::PRINTF("var_index :");
                // for(int32_t i=0; i<totaldims[0]; ++i){
                //     AscendC::PRINTF("%d ", var_index[i]);
                // }
                // // AscendC::PRINTF("\n");
                // AscendC::PRINTF("updates_index :");
                // for(int32_t i=0; i<totaldims[2]; ++i){
                //     AscendC::PRINTF("%d ", updates_index[i]);
                // }
                // AscendC::PRINTF("\n");

                // get var flat_index
                int32_t flat_index_var = 0;
                stride = 1;
                for(int32_t k = totaldims[0] -1; k >= 0; --k){
                    flat_index_var += var_index[k] * stride;
                    stride *= shape[0][k];
                } 
                // get updates flat_index
                int32_t flat_index_updates = 0;
                stride = 1;
                for(int32_t k = totaldims[2] -1; k >= 0; --k){
                    flat_index_updates += updates_index[k] * stride;
                    stride *= shape[2][k];
                }

                // AscendC::PRINTF("flat_index_updates %d, flat_index_var %d\n", flat_index_updates, flat_index_var);
                // update var value according to mode           
                F a1 = Gm_updates.GetValue(flat_index_updates);
                F a2 = Gm_var.GetValue(flat_index_var);
                
                // AscendC::PRINTF("a1 %f\n", a1);
                // AscendC::PRINTF("a2 %f\n", a2);
                if(this->mode == 2){
                    auto result = a1 + a2;
                    // AscendC::PRINTF("result %f\n", result);
                    Gm_var.SetValue(flat_index_var, (T)result);
                }else if(this->mode == 3){
                    auto result = a1 * a2;
                    // AscendC::PRINTF("result %f\n", result);
                    Gm_var.SetValue(flat_index_var, (T)result);
                }else{
                    auto result = a1;
                    // AscendC::PRINTF("result %f\n", result);
                    Gm_var.SetValue(flat_index_var, (T)result);
                }
            }      
        }else{
            for(int32_t flat_index = 0; flat_index < this->tile_length ; ++flat_index){
                // AscendC::PRINTF("flat_index %d\n", flat_index);
                int32_t flat_index_true = flat_index_begin + flat_index; // indices_index
                
                // get updates_index according to flat_index_begin
                int32_t stride = 1;
                for(int32_t k = totaldims[1] - 1; k >= 0; --k){
                    updates_index[k] = (flat_index_true / stride) % shape[1][k];
                    var_index[k] = updates_index[k];
                    stride *= shape[1][k];
                }
                // update var_index according to updates_index
                // for(int32_t k = 0; k < totaldims[0] ; ++k){
                //     var_index[k] = updates_index[k];
                // } 
                // AscendC::PRINTF("flat_index %d\n", flat_index);
                // AscendC::printf("flat_index %d\n", flat_index);

                // update var index
                TYPE_INDICES new_idx = Gm_indices.GetValue(flat_index);
                var_index[axis] = new_idx;

                // 打印 var_index 和 updates_index`
                // AscendC::PRINTF("var_index :");
                // for(int32_t i=0; i<totaldims[0]; ++i){
                //     AscendC::PRINTF("%d ", var_index[i]);
                // }
                // // AscendC::PRINTF("\n");
                // AscendC::PRINTF("updates_index :");
                // for(int32_t i=0; i<totaldims[2]; ++i){
                //     AscendC::PRINTF("%d ", updates_index[i]);
                // }
                // AscendC::PRINTF("\n");

                // get var flat_index
                int32_t flat_index_var = 0;
                stride = 1;
                for(int32_t k = totaldims[0] -1; k >= 0; --k){
                    flat_index_var += var_index[k] * stride;
                    stride *= shape[0][k];
                } 
                // AscendC::PRINTF("flat_index_updates %d, flat_index_var %d\n", flat_index_updates, flat_index_var);
                // update var value according to mode           
                F a1 = Gm_updates.GetValue(0);
                F a2 = Gm_var.GetValue(flat_index_var);
                
                // AscendC::PRINTF("a1 %f\n", a1);
                // AscendC::PRINTF("a2 %f\n", a2);
                if(this->mode == 2){
                    auto result = a1 + a2;
                    // AscendC::PRINTF("result %f\n", result);
                    Gm_var.SetValue(flat_index_var, (T)result);
                }else if(this->mode == 3){
                    auto result = a1 * a2;
                    // AscendC::PRINTF("result %f\n", result);
                    Gm_var.SetValue(flat_index_var, (T)result);
                }else{
                    auto result = a1;
                    // AscendC::PRINTF("result %f\n", result);
                    Gm_var.SetValue(flat_index_var, (T)result);
                }
            }
        }

    }
    __aicore__ inline void Broadcast() {
        // 广播
        // src 与 self 之间的广播：    
 
    }
private:
    __aicore__ inline void ResovleTiling(
        uint32_t total_length, uint32_t tile_num_mean, uint32_t tile_num_end,
        uint32_t tile_length_mean, uint32_t tile_length_end) {
        uint32_t pad32 = BLOCK_SIZE;  // 对齐32B需要的最小数据量
        this->total_length = total_length;
        this->block_id = AscendC::GetBlockIdx();
        if (AscendC::GetBlockNum() > 1 && AscendC::GetBlockIdx() == (AscendC::GetBlockNum() - 1)) {
            if(tile_num_end == 0){
                this->tile_length = tile_length_mean;
            }else{
                this->tile_length = tile_length_end;
            }
        }else {
            this->tile_length = tile_length_mean;
        }
        this->block_length = tile_length_mean;      
        // AscendC::PRINTF("total_length %d\n", total_length);
        // AscendC::PRINTF("tile_length_mean %d\n", tile_length_mean);  
    }
private:
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<TYPE_INDICES> Gm_indices;
    AscendC::GlobalTensor<TYPE_VAR> Gm_var;
    AscendC::GlobalTensor<TYPE_UPDATES> Gm_updates;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> x1_inque,x2_inque;
    int32_t shape[3][64];
    int32_t totalSize[3] = {1,1,1};
    int32_t totaldims[3];
    int32_t var_index[64]; // 记录var对应每个索引
    int32_t updates_index[64]; // 记录updates和indices 对应的每个索引(其中updates的索引与indices的索引相同)
    int32_t axis;
    int32_t mode;
    int32_t var_length;
    int32_t updates_length;
    int32_t indices_length;
    uint32_t total_length, block_length, tile_length, block_id;
    // __aicore__ inline void BroadcastShapes() {
    //     // Ensure updates and indices shapes match var
    //     for (int32_t i = 0; i < totaldims[0]; ++i) {
    //         if (shape[1][i] == 1 && shape[0][i] > 1) {
    //             shape[1][i] = shape[0][i];
    //         }
    //         if (shape[2][i] == 1 && shape[0][i] > 1) {
    //             shape[2][i] = shape[0][i];
    //         }
    //         ASSERT((shape[1][i] == shape[0][i] || shape[1][i] == 1) && "Broadcast for indices failed!");
    //         ASSERT((shape[2][i] == shape[0][i] || shape[2][i] == 1) && "Broadcast for updates failed!");
    //     }
    // }
};
extern "C" __global__ __aicore__ void scatter_elements(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelScatterElements<DTYPE_VAR, DTYPE_INDICES, DTYPE_UPDATES> op;
    op.Init(var, indices, updates, tiling_data.axis, tiling_data.mode,
    tiling_data.shape, tiling_data.size, tiling_data.ndims,
    tiling_data.totalLength, tiling_data.tileNumMean,
    tiling_data.tileNumEnd, tiling_data.tileLengthMean,
    tiling_data.tileLengthEnd);
    // op.Process();
    op.Process_muti_core();
}
