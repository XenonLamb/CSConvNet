int CSKernelConv2D_forward_cuda(
	at::Tensor& input,
	at::Tensor& kernel_bank,
    int kernel_size,
	at::Tensor& output,
	at::Tensor& buckets
);

int CSKernelConv2D_backward_cuda(
	at::Tensor& input,
	at::Tensor& kernel_bank,
    int kernel_size,
	at::Tensor& grad_output,
	at::Tensor& grad_input,
	at::Tensor& grad_kernel,
	at::Tensor& buckets
);