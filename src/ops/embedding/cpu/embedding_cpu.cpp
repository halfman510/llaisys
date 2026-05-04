#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

// #include <cstdint>
/*
index: 1-D，长度 num_rows
weight: 2-D，形状 (vocab_size, emb_dim)
out: 2-D，形状 (num_rows, emb_dim)
*/

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t num_rows, size_t emb_dim) {
    for (size_t i = 0; i < num_rows; ++i) {//num_rows是index的总长度
        int64_t row = index[i];//这里的index应该是输入token对应的token id
        for(size_t j = 0; j < emb_dim; ++j){
            out[i * emb_dim + j] = weight[row * emb_dim + j];//复制一整行，实际上还是一维方式存储的
        }
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t num_rows, size_t emb_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), 
                          reinterpret_cast<const int64_t *>(index), 
                          reinterpret_cast<const float *>(weight), 
                          num_rows, emb_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), 
                          reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const llaisys::bf16_t *>(weight), 
                          num_rows, emb_dim);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), 
                          reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const llaisys::fp16_t *>(weight), 
                          num_rows, emb_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
