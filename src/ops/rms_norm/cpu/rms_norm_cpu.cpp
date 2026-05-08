#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

// #include <cstdint>
#include <cmath>
/*
思路：1.按行做归一化，求出这一行的 rms 大小 2.这行每个元素除以 rms，再乘上对应位置的 weight
*/

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t rows, size_t cols, float eps) {//rows表示输入多少行，cols表示每行多少列
    for(size_t i = 0; i < rows; ++i){//遍历每一行
        const T *x = in + i * cols;
        T *y = out + i * cols;//找到当前行的起始地址

        float sum_sq = 0.f;
        for(size_t j = 0; j < cols; ++j){//计算这行所有元素的平方和
            float xv = llaisys::utils::cast<float>(x[j]);
            sum_sq += xv * xv;
        }

        float scale = 1.0f / std::sqrt(sum_sq / static_cast<float>(cols) + eps);//算出缩放因子，一会儿每行直接乘这个缩放因子即可

        for(size_t j = 0; j < cols; ++j){
            float xv = llaisys::utils::cast<float>(x[j]);
            float wv = llaisys::utils::cast<float>(weight[j]);
            y[j] = llaisys::utils::cast<T>(xv * scale * wv);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type, size_t rows, size_t cols, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), 
                          reinterpret_cast<const float *>(in), 
                          reinterpret_cast<const float *>(weight), 
                          rows, cols, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), 
                          reinterpret_cast<const llaisys::bf16_t *>(in), 
                          reinterpret_cast<const llaisys::bf16_t *>(weight), 
                          rows, cols, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), 
                          reinterpret_cast<const llaisys::fp16_t *>(in), 
                          reinterpret_cast<const llaisys::fp16_t *>(weight), 
                          rows, cols, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
