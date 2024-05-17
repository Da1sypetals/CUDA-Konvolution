#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void leg2d_launcher(const float *x, float *leg,
                    int batch_size, int in_channels, int height, int width, int degree);
void leg2d_bwd_launcher(const float *gout, const float *x, const float *leg, float *grad_x,
                        int batch_size, int in_channels, int width, int height, int degree);

torch::Tensor leg2d_cuda_fwd(torch::Tensor x, int degree)
{

    CHECK_INPUT(x);

    const float *x_ptr = x.data_ptr<float>();
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);

    // create leg tensor
    torch::Tensor leg = torch::ones({batch_size, ((degree + 1) * in_channels), height, width},
                                    torch::device(torch::kCUDA).dtype(torch::kFloat));

    float *leg_ptr = leg.data_ptr<float>();

    leg2d_launcher(x_ptr, leg_ptr, batch_size, in_channels, height, width, degree);

    return leg;
}

torch::Tensor leg2d_cuda_bwd(torch::Tensor gout, torch::Tensor x, torch::Tensor leg, int degree)
{

    CHECK_INPUT(x);
    CHECK_INPUT(leg);

    const float *gout_ptr = gout.data_ptr<float>();
    const float *x_ptr = x.data_ptr<float>();
    const float *leg_ptr = leg.data_ptr<float>();

    int batch_size = x.size(0);
    int in_channels = x.size(1) / degree;
    int height = x.size(2);
    int width = x.size(3);

    // create grad_x tensor
    torch::Tensor grad_x = torch::zeros_like(x);
    float *grad_x_ptr = grad_x.data_ptr<float>();

    leg2d_bwd_launcher(gout_ptr, x_ptr, leg_ptr, grad_x_ptr, batch_size, in_channels, height, width, degree);

    return grad_x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &leg2d_cuda_fwd, "leg2d forward");
    m.def("backward", &leg2d_cuda_bwd, "leg2d backward");
}