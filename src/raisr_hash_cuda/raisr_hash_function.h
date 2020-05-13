#ifdef __cplusplus
	extern "C" {
#endif


int raisr_compute_grads_kernel(
	at::Tensor& in_GX,
	at::Tensor& in_GY,
    int hsize,
    int row,
    int col,
	at::Tensor& tensor_phi,
	at::Tensor& tensor_lambda,
	at::Tensor& tensor_mu,
	cudaStream_t stream
);



#ifdef __cplusplus
	}
#endif
