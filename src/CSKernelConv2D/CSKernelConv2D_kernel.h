#ifdef __cplusplus
	extern "C" {
#endif

int CSKernelConv2D_forward_cuda_kernel(
	at::Tensor& input,
	at::Tensor& kernel_bank,
	int kernel_size,
	at::Tensor& output,
	at::Tensor& buckets,
	cudaStream_t stream
);

int CSKernelConv2D_backward_cuda_kernel(
	at::Tensor& input,
	at::Tensor& kernel_bank,
	int kernel_size,
	at::Tensor& grad_output,
	at::Tensor& grad_input,
	at::Tensor& grad_kernel,
	at::Tensor& buckets,
	cudaStream_t stream
);


#ifdef __cplusplus
	}
#endif