#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void leg_launcher(const torch::PackedTensorAccessor64<float, 2> x, 
                    torch::PackedTensorAccessor64<float, 3> leg, 
                    int batch_size, int in_feats, int degree);
void leg_bwd_launcher(const torch::PackedTensorAccessor64<float, 3> gout, 
                    const torch::PackedTensorAccessor64<float, 2> x, 
                    const torch::PackedTensorAccessor64<float, 3> leg, 
                    torch::PackedTensorAccessor64<float, 2> grad_x, 
                    int batch_size, int in_feats, int degree);

torch::Tensor leg_cuda_fwd(torch::Tensor x, int degree)
{

    CHECK_INPUT(x);

    const auto x_acc = x.packed_accessor64<float, 2>();
    int batch_size = x.size(0);
    int in_feats = x.size(1);

    // create leg tensor
    torch::Tensor leg = torch::ones({degree + 1, batch_size, in_feats},
                                    torch::device(torch::kCUDA).dtype(torch::kFloat));

    auto leg_acc = leg.packed_accessor64<float, 3>();

    leg_launcher(x_acc, leg_acc, batch_size, in_feats, degree);

    cudaDeviceSynchronize();

    return leg;
}

torch::Tensor leg_cuda_bwd(torch::Tensor gout, torch::Tensor x, torch::Tensor leg)
{

    CHECK_INPUT(x);
    CHECK_INPUT(leg);

    const auto gout_acc = gout.packed_accessor64<float, 3>();
    const auto x_acc = x.packed_accessor64<float, 2>();
    const auto leg_acc = leg.packed_accessor64<float, 3>();

    int batch_size = x.size(0);
    int in_feats = x.size(1);
    int degree = leg.size(0) - 1;

    // create grad_x tensor
    torch::Tensor grad_x = torch::zeros_like(x);
    auto grad_x_acc = grad_x.packed_accessor64<float, 2>();

    leg_bwd_launcher(gout_acc, x_acc, leg_acc, grad_x_acc, batch_size, in_feats, degree);

    cudaDeviceSynchronize();

    return grad_x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &leg_cuda_fwd, "leg forward");
    m.def("backward", &leg_cuda_bwd, "leg backward");
}