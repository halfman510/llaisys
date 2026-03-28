#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();//保存维度数量
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {//累计计算得到总元素数(从最低维度开始计算，算每层stride步长 * 上一层shape形状)
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);//获取单个元素有多少字节

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {//请求CPU张量，但当前上下文设备不是CPU
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);//对CPU来说等价host，GPU是设备内存
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {//指向tensor的起始位置
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {//同上，const常量不可修改，可以用来读
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {//返回tensor维度
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {//返回形状（各维度大小），比如{2, 3, 4}是一个三维tensor
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {//返回数据枚举类型
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {//CPU还是GPU
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {//返回总元素数，shape维度累乘，{2,3} 返回 6
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {//输出格式化描述字符串
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    const auto &shape_ = this->shape();//auto编译器自动推导变量，避免手写类型和返回类型不一样
    const auto &strides_ = this->strides();

    if(shape_.size() != strides_.size()) {
        return false;
    }
    if(this->numel() == 0){
        return false;
    }

    ptrdiff_t expected = 1;//ptrdiff_t指针差值类型，有符号整数，用于可能出现负数的场景，比如： stride、索引差
    for (size_t i = shape_.size(); i-- > 0;){//和(size_t i = n; i > 0; i--)不等价，第一种循环体内取值为n-1 ... 0（判断条件时就-1），第二种循环体内取值为n ... 1（判断条件只比较不改值，每轮末尾执行）
        if (strides_[i] != expected) {
            return false;
        }
        expected *= static_cast<ptrdiff_t>(shape_[i]);//static_cast显式类型转换
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    const size_t n = this->ndim();
    // 1.检查改变前后维度是否相同
    CHECK_ARGUMENT(
        order.size() == n,
        "permute order size must equel to tensor ndim");

    // 2.检查 order 是否合法，order 必须是[0,n)的一个不重复不越界的排列
    std::vector<bool> seen(n, false);
    for(size_t i = 0; i < n; i++){
        const size_t axis = order[i];
        CHECK_ARGUMENT(axis < n, "permute order contains out-of-range axis");
        CHECK_ARGUMENT(!seen[axis], "permute order contains duplicate axis");
        seen[axis] = true;
    }

    // 3.根据 order 重排 shape 和 stride
    std::vector<size_t> new_shape(n);
    std::vector<ptrdiff_t> new_strides(n);//这里类型自适应，否则容易搞不对类型报错
    for(size_t i = 0; i < n; i++){
        new_shape[i] = _meta.shape[order[i]];//_meta是当前this对象的私有成员变量，直接用就可以访问
        new_strides[i] = _meta.strides[order[i]];//strides 描述数据在内存中的真实布局，因此这里不用按照连续的来设置 strides
    }

    // 4.共享 storage,不拷贝数据，保留 offset
    TensorMeta new_meta{_meta.dtype, std::move(new_shape), std::move(new_strides)};//C++列表初始化，针对聚合类型（struch，公开成员、无自定义构造函数）

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {//只有连续张量，才可以通过拆分或合并改变形状，不连续张量不可以
    //1.元素总数一致
    const size_t old_numel = this->numel();
    const size_t new_numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());//multiplies<size_t>，size_t确定参与乘法的类型
    
    CHECK_ARGUMENT(
        old_numel == new_numel, 
        "view shape is incompatible with input tensor numel");

    //2.是否连续
    CHECK_ARGUMENT(
        this->isContiguous(), 
        "view requires contiguous input tensor");

    //3.创建新shape的新stride
    std::vector<ptrdiff_t> new_strides(shape.size(), 0);//新步长vector长度 = shape.size()shape维度，后面的0是填充每个元素初值为0
    ptrdiff_t stride = 1;
    for(size_t i = shape.size(); i-- > 0;){
        new_strides[i] = stride;
        stride *= static_cast<ptrdiff_t>(shape[i]);
    }

    //4.共享storage
    TensorMeta new_meta{this->dtype(), shape, std::move(new_strides)};//shape传的是拷贝，new_strides是局部变量，后面再也不需要，move比拷贝开销更小，move后原来的new_strides为空
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {//在第 dim 个维度上，取 [start, end) 这一段
    // 1.检查参数
    CHECK_ARGUMENT(dim < this->ndim(), "slice dim out of range");
    CHECK_ARGUMENT(start <= end, "slice requires start <= end");
    CHECK_ARGUMENT(end < _meta.shape[dim], "slice end out of range");

    // 2.修改 shape
    std::vector<size_t> new_shape = _meta.shape;//因为函数后面加了const，所以不能修改当前对象 this
    new_shape[dim] = end - start;

    // 3.stride 不变
    std::vector<ptrdiff_t> new_strides = _meta.strides;

    // 4.offset
    size_t new_offset = _offset + start * static_cast<size_t>(_meta.strides[dim]) * this->elementSize();//强制类型转换，是因为这里有的有符号、有的无符号，混算有隐式转换，还容易有编译警告

    // 5.构造新meta
    TensorMeta new_meta{this->dtype(), std::move(new_shape), std::move(new_strides)};
    return std::shared_ptr<Tensor>(new Tensor(std::move(new_meta), this->_storage, new_offset));
    // return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage));
}

void Tensor::load(const void *src_) {//把主机端的数据，复制到张量storage对应位置中
    const size_t bytes = this->numel() * this->elementSize();
    if(bytes == 0) {
        return;
    }
    CHECK_ARGUMENT(src_ != nullptr, "source pointer must not be null");

    core::context().setDevice(this->deviceType(), this->deviceId());
    const llaisysMemcpyKind_t memcpy_kind = (this->deviceType() == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D;
    core::context().runtime().api()->memcpy_sync(this->data(), src_, bytes, memcpy_kind);
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
