//#include <torch/torch.h>
//#include <torch/extension.h>
//#include <ATen/ATen.h>
//#include <ATen/Context.h>
//#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <ATen/ATen.h>

#include "KernelConv2D_kernel.h"


int CSKernelConv2D_forward_cuda(
	at::Tensor& input,
	at::Tensor& kernel_bank,
    int kernel_size,
	at::Tensor& output,
	at::Tensor& buckets
) {
    int success = KernelConv2D_forward_cuda_kernel(
        input,
        kernel_bank,
        kernel_size,
        output,
        buckets,
        at::cuda::getCurrentCUDAStream()
    );
    if (!success) {
    	AT_ERROR("CUDA call failed");
    }
	return 1;
}

int CSKernelConv2D_backward_cuda(
	at::Tensor& input,
	at::Tensor& kernel_bank,
    int kernel_size,
	at::Tensor& grad_output,
	at::Tensor& grad_input,
	at::Tensor& grad_kernel,
	at::Tensor& buckets
) {

    int success = KernelConv2D_backward_cuda_kernel(
        input,
        kernel_bank,
        kernel_size,
        grad_output,
        grad_input,
        grad_kernel,
        buckets,
        at::cuda::getCurrentCUDAStream()
    );
    if (!success) {
    	AT_ERROR("CUDA call failed");
    }
	return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &CSKernelConv2D_forward_cuda, "KernelConv2D forward (CUDA)");
    m.def("backward", &CSKernelConv2D_backward_cuda, "KernelConv2D backward (CUDA)");
}


