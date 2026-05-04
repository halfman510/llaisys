#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);

    CHECK_ARGUMENT(index->ndim() == 1, "Embedding: index must be 1D");
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index dtype must be i64");
    CHECK_ARGUMENT(weight->ndim() == 2, "Embedding: weight must be 2D");
    CHECK_ARGUMENT(out->ndim() == 2, "Embedding: out must be 2D");
    CHECK_ARGUMENT(out->shape()[0] == index->numel(), "Embedding: out rows must match index length");
    CHECK_ARGUMENT(out->shape()[1] == weight->shape()[1], "Embedding: out cols must match weight cols");
    CHECK_ARGUMENT(out->dtype() == weight->dtype(), "Embedding: out dtype must match weight dtype");

    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
            "Embedding: all tensors must be contiguous.");
    
    if(out->deviceType() == LLAISYS_DEVICE_CPU){
        return cpu::embedding(out->data(), index->data(), weight->data(),
                              out->dtype(), index->numel(), weight->shape()[1]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()){
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(),
                              out->dtype(), index->numel(), weight->shape()[1]);
    
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
    // TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops
