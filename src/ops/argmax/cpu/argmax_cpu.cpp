#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
// #include <cstdint>
// #include <type_traits>

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    // 约定调用前已保证 numel > 0
    size_t best_idx = 0;
    T best_val = vals[0];//vals是张量首地址

    for(size_t i = 1; i < numel; i++){//遍历所有元素获得最大值
        bool greater = false;

        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>){
            greater = llaisys::utils::cast<float>(vals[i]) > llaisys::utils::cast<float>(best_val);
        } else{
            greater = vals[i] > best_val;
        }

        if(greater){
            best_idx = i;
            best_val = vals[i];
        }
    }

    max_idx[0] = static_cast<int64_t>(best_idx);//[0]是因为当前约定max_idx和max_val都是1D张量，以后扩展为多维张量时候返回max_dix[i]
    max_val[0] = best_val;
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<float *>(max_val), 
                    reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::bf16_t *>(max_val),
                    reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::fp16_t *>(max_val),
                    reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
