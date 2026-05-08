#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);

    CHECK_ARGUMENT(out->ndim() == 2, "RmsNorm: out must be 2D");
    CHECK_ARGUMENT(in->ndim() == 2, "RmsNorm: in must be 2D");
    CHECK_ARGUMENT(weight->ndim() == 1, "RmsNorm: weight must be 1D");

    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0], "RmsNorm: row mismatch");
    CHECK_ARGUMENT(out->shape()[1] == in->shape()[1], "RmsNorm: col mismatch");
    CHECK_ARGUMENT(weight->shape()[0] == in->shape()[1], "RmsNorm: weight size mismatch");

    CHECK_ARGUMENT(out->dtype() == in->dtype(), "RmsNorm: dtype mismatch");
    CHECK_ARGUMENT(out->dtype() == weight->dtype(), "RmsNorm: dtype mismatch");
    CHECK_ARGUMENT(eps >= 0.f, "RmsNorm: eps must be non-negative");

    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "RmsNorm: all tensors must be contiguous.");

    if(out->deviceType() == LLAISYS_DEVICE_CPU){
        return cpu::rms_norm(out->data(), in->data(), weight->data(),
                            out->dtype(), in->shape()[0], in->shape()[1], eps);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()){
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(),
                            out->dtype(), in->shape()[0], in->shape()[1], eps);
#ifdef ENABLE_NVIDIA_API
        case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
