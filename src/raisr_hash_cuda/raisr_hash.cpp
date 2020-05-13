#include <torch/torch.h>
#include <ATen/ATen.h>
//#include <ATen/cuda/CUDAContext.h>

//#include <torch/torch.h>
//#include <torch/extension.h>
//#include <ATen/ATen.h>
#include <ATen/Context.h>
//#include <ATen/cuda/CUDAContext.h>
#include "raisr_hash_function.h"



int raisr_hash(
	at::Tensor& in_GX,
	at::Tensor& in_GY,
    int hsize,
    int row,
    int col,
	at::Tensor& tensor_phi,
	at::Tensor& tensor_lambda,
	at::Tensor& tensor_mu
) {
    int success = raisr_compute_grads_kernel(
        in_GX,
        in_GY,
        hsize,
        row,
        col,
        tensor_phi,
        tensor_lambda,
        tensor_mu,
        //at::cuda::getCurrentCUDAStream()
        at::globalContext().getCurrentCUDAStream()
    );
    if (!success) {
    	AT_ERROR("CUDA call failed");
    }
	return 1;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &raisr_hash, "raisr classification(CUDA)");
}
