#include "linear_cpu.hpp"

#include "../../../utils.hpp"

// #include <cstdint>
/*
in 形状：(batch, in_features)
weight 形状：(out_features, in_features)（未转置存储）
out 形状：(batch, out_features)
bias 形状：out_features
*/

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t batch, size_t out_features, size_t in_features) {//矩阵乘
    for(size_t i = 0; i < batch; ++i){//站在输出矩阵的角度,先行后列，然后对out要输出的每个元素求行列点积
        for(size_t o = 0; o < out_features; ++o){
            float acc = 0.f;
            for(size_t k = 0; k < in_features; ++k){
                acc += llaisys::utils::cast<float>(in[i * in_features + k]) *
                        llaisys::utils::cast<float>(weight[o * in_features + k]);
            }
            if(bias) acc += llaisys::utils::cast<float>(bias[o]);
            out[i * out_features + o] = llaisys::utils::cast<T>(acc);
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, size_t batch, size_t out_features, size_t in_features) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), 
                          reinterpret_cast<const float *>(in), 
                          reinterpret_cast<const float *>(weight), 
                          reinterpret_cast<const float *>(bias),
                          batch, out_features, in_features);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), 
                          reinterpret_cast<const llaisys::bf16_t *>(in), 
                          reinterpret_cast<const llaisys::bf16_t *>(weight), 
                          reinterpret_cast<const llaisys::bf16_t *>(bias),
                          batch, out_features, in_features);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), 
                          reinterpret_cast<const llaisys::fp16_t *>(in), 
                          reinterpret_cast<const llaisys::fp16_t *>(weight), 
                          reinterpret_cast<const llaisys::fp16_t *>(bias),
                          batch, out_features, in_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
