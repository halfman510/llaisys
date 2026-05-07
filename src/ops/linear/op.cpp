#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"


namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if(bias){
        CHECK_SAME_DEVICE(out, bias);
    }

    CHECK_ARGUMENT(out->ndim() == 2, "Linear: out must be 2D");
    CHECK_ARGUMENT(in->ndim() == 2, "Linear: in must be 2D");
    CHECK_ARGUMENT(weight->ndim() == 2, "Linear: weight must be 2D");
    if(bias){
        CHECK_ARGUMENT(bias->ndim() == 1, "Linear: bias must be 1D");
    }

    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0], "Linear: out batch mismatch");
    CHECK_ARGUMENT(out->shape()[1] == weight->shape()[0], "Linear: out features mismatch");
    CHECK_ARGUMENT(in->shape()[1] == weight->shape()[1], "Linear: in features mismatch");
    if(bias){
        CHECK_ARGUMENT(bias->shape()[0] == weight->shape()[0], "Linear: bias size mismatch");
    }

    CHECK_ARGUMENT(out->dtype() == in->dtype(), "Linear: dtype mismatch");
    CHECK_ARGUMENT(out->dtype() == weight->dtype(), "Linear: dtype mismatch");
    if(bias){
        CHECK_ARGUMENT(out->dtype() == bias->dtype(), "Linear: dtype mismatch");
    }

    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous() && (!bias || bias->isContiguous()),
    "Linear: all tensors must be contiguous.");

    if(out->deviceType() == LLAISYS_DEVICE_CPU){
        return cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data():nullptr,
        out->dtype(), in->shape()[0], out->shape()[1], in->shape()[1]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()){
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data():nullptr,
        out->dtype(), in->shape()[0], out->shape()[1], in->shape()[1]);
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
