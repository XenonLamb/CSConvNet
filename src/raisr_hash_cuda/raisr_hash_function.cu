//Update the implemention to support 2D kernel by Shangchen Zhou
//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <ATen/ATen.h>
//#include <ATen/Context.h>
//#include <ATen/cuda/CUDAContext.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>

#define pi 3.141592653589793

#define THREAD_PER_BLOCK 512

#define VEC_0(ARRAY) ((ARRAY).x)
#define VEC_1(ARRAY) ((ARRAY).y)
#define VEC_2(ARRAY) ((ARRAY).z)
#define VEC_3(ARRAY) ((ARRAY).w)

#define IDX_1(ARRAY, X)          ((ARRAY)[((X) * (ARRAY##_stride.x))])
#define IDX_2(ARRAY, X, Y)       ((ARRAY)[((X) * (ARRAY##_stride.x)) + ((Y) * (ARRAY##_stride.y))])
#define IDX_3(ARRAY, X, Y, Z)    ((ARRAY)[((X) * (ARRAY##_stride.x)) + ((Y) * (ARRAY##_stride.y)) + ((Z) * (ARRAY##_stride.z))])
#define IDX_4(ARRAY, X, Y, Z, W) ((ARRAY)[((X) * (ARRAY##_stride.x)) + ((Y) * (ARRAY##_stride.y)) + ((Z) * (ARRAY##_stride.z)) + ((W) * (ARRAY##_stride.w))])



#ifdef __cplusplus
	extern "C" {
#endif

//Define forward operations

// input and output should be of shape [batch_size, n_features, H,W]
// kernel should be of shape [batch_size, n_features*n_features*kernel_size*kernel_size, H, W]



__global__ void raisr_hash_cuda(int n_output, float *in_GX, float *in_GY, int hsize,int row, int col, float* tensor_phi, float* tensor_lambda, float* tensor_mu)
{
    //int x = threadIdx.x+ blockIdx.x* blockDim.x;
    //int y = threadIdx.y+ blockIdx.y* blockDim.y;

    int intIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (intIndex >= n_output) {
		return;
	}
    int y = intIndex / col;
    int x = intIndex % col;

    int p_hsize = hsize;
    //int tid = j1 + i1*col;
    if(x < p_hsize ||x >= col - p_hsize)
        return;
    if(y < p_hsize ||y >= row - p_hsize)
        return;

    int pos = y*col+x;

    //float dx_val=0.0;
    //float dy_val=0.0;
    //int rect_w=2*p_hsize+1;
    //int rect_h=2*p_hsize+1;

    float g00 = 0.0;
    float g01 = 0.0;
    float g11 = 0.0;


    //for i in range(0-p_hsize,p_hsize+1):
    //    for j in range(0-p_hsize,p_hsize+1):
    //        dx_val = dx[(y+i),(x+j)]
    //        dy_val = dy[(y+i),(x+j)]
    //        g00 += dx_val * dx_val
    //        g01 += dx_val * dy_val
    //        g11 += dy_val * dy_val

    for(int i = 0-p_hsize;i<p_hsize+1;i++)
    {
        for(int j= 0-p_hsize;j<p_hsize+1;j++)
        {
            float dx_val = in_GX[(y+i)*col+x+j];
            float dy_val = in_GY[(y+i)*col+x+j];
            g00 += dx_val * dx_val;
            g01 += dx_val * dy_val;
            g11 += dy_val * dy_val;
        }
    }

    //add normalize to be consistent with former code
    g00 = g00 / ((float)((2*p_hsize+1)*(2*p_hsize+1)));
    g01 = g01 / ((float)((2*p_hsize+1)*(2*p_hsize+1)));
    g11 = g11 / ((float)((2*p_hsize+1)*(2*p_hsize+1)));


    float tmp1 = g00 + g11;
    float tmp2 = sqrt((g00 - g11) * (g00 - g11) + 4 * g01 * g01);
    float S1 = (tmp1 + tmp2) / 2;
    float S2 = (tmp1 - tmp2) / 2;

    float theta = 0;

    if(fabs(g01)>1e-9)
    {
        //theta =atan((g00-S1)/(-g01))/pi*180.0;
        theta =atan((S1-g00)/(g01))/pi*180.0;
    }
    else if(g00>g11)
    {
        theta = 90;
    }
    else{
        theta = 0;
    }

    //theta =atan((S1-g00)/(g01))/pi*180.0;
    if(theta<0){
        theta = theta + 180.0;
    }
    theta = theta / 180.0;


    //int Q_theta = ceil(theta/180*Qangle);
    float lamda = sqrt(fabs(S1));

    float u = (sqrt(fabs(S1)) - sqrt(fabs(S2)))/(sqrt(fabs(S1)) + sqrt(fabs(S2)) + 0.00000000000000001);
    tensor_phi[pos] = theta;
    tensor_lambda[pos] = lamda;
    tensor_mu[pos] = u;


};

/*
__global__ void remosaic_cuda(float *quad_raw, float *raw, float *in_GX, float *in_GY,int h_hsize,int f_hsize,int types_size, int Qangle, int Qstrength, int Qcoherence,  float *stre,  float * cohe, float *Q, float *V,float *mark,int row,int col,int width)
{
    int x = threadIdx.x+ blockIdx.x* blockDim.x;
    int y = threadIdx.y+ blockIdx.y* blockDim.y;

    int p_hsize = max(h_hsize, f_hsize);
    //int tid = j1 + i1*col;
    if(x < p_hsize ||x >= col - p_hsize)
        return;
    if(y < p_hsize ||y >= row - p_hsize)
        return;


    int pos = y % types_size * types_size + x % types_size;


    float dx_val=0.0;
    float dy_val=0.0;
    int rect_w=2*p_hsize+1;
    int rect_h=2*p_hsize+1;

    float g00 = 0.0;
    float g01 = 0.0;
    float g11 = 0.0;


    //for i in range(0-p_hsize,p_hsize+1):
    //    for j in range(0-p_hsize,p_hsize+1):
    //        dx_val = dx[(y+i),(x+j)]
    //        dy_val = dy[(y+i),(x+j)]
    //        g00 += dx_val * dx_val
    //        g01 += dx_val * dy_val
    //        g11 += dy_val * dy_val

    for(int i = 0-p_hsize;i<p_hsize+1;i++)
    {
        for(int j= 0-p_hsize;j<p_hsize+1;j++)
        {
            float dx_val = in_GX[(y+i)*width+x+j];
            float dy_val = in_GY[(y+i)*width+x+j];
            g00 += dx_val * dx_val;
            g01 += dx_val * dy_val;
            g11 += dy_val * dy_val;
        }
    }

    float tmp1 = g00 + g11;
    float tmp2 = sqrt((g00 - g11) * (g00 - g11) + 4 * g01 * g01);
    float S1 = (tmp1 + tmp2) / 2;
    float S2 = (tmp1 - tmp2) / 2;

    float theta = 0;
    if(fabs(g01)>1e-9)
    {
        theta =atan((g00-S1)/(-g01))/pi*180;
    }
    else if(g00>g11)
    {
        theta = 90;
    }
    else{
        theta = 0;
    }

    if(theta<0){
        theta = theta + 180;
    }


    int Q_theta = ceil(theta/180*Qangle);
    if(Q_theta==0){
         Q_theta = 1;
    }
    Q_theta= Q_theta -1;


    float lamda = sqrt(S1);
    float u = (sqrt(S1) - sqrt(S2))/(sqrt(S1) + sqrt(S2) + 0.00000000000000001);

    int Q_lamda = Qstrength-1;
	int Q_u = Qcoherence -1;
	for(int k=Qstrength-1; k>=0; k--){
		Q_lamda = lamda < stre[k] ? k : Q_lamda;
	}
	for(int k=Qcoherence-1; k>=0; k--){
		Q_u = u < cohe[k] ? k : Q_u;
	}


    int dim0 = Qangle*Qstrength * Qcoherence;
    int index = Q_theta * Qstrength * Qcoherence + Q_lamda * Qcoherence + Q_u;

    atomicAdd(mark + pos*dim0+index, 1);


    int patchL_shape_2 = (2*h_hsize+1)*(2*h_hsize+1);
    int V_offset = pos*dim0*patchL_shape_2+index*patchL_shape_2;
    //int Q_offset = pos*dim0*patchL_shape_2*patchL_shape_2 + index*patchL_shape_2*patchL_shape_2;
    int Q_offset = V_offset*patchL_shape_2;

    float *Q_ptr = Q+Q_offset;
    float *V_ptr = V+V_offset;

    int patchL_dim0 = 0;
    int patchL_dim1 = 0;

    for(int i=y-h_hsize;i<y+h_hsize+1;i++)
    {
        for(int j=x-h_hsize; j< x+h_hsize+1;j++)
        {
            float val = quad_raw[i*width+j];
            for(int k= y-h_hsize;k<y+h_hsize+1;k++)
            {
                for(int m= x-h_hsize;m<x+h_hsize+1;m++)
                {
                    float ret=val*quad_raw[k*width+m];
                    atomicAdd(Q_ptr + patchL_dim1, ret);
                    patchL_dim1+=1;
                }
            }
            float b1 = val*raw[y*width+x];
            atomicAdd(V_ptr+patchL_dim0, b1);
            patchL_dim0 +=1;
        }
    }

};





int KernelConv2D_forward_cuda_kernel(
	at::Tensor& input,
	at::Tensor& kernel,
	int kernel_size,
	at::Tensor& output,
	cudaStream_t stream
) {
	int n_output = 0;
	n_output = output.size(0) * output.size(1) * output.size(2) * output.size(3);
	KernelConv2D_forward_function<<< (n_output + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK, 0, stream >>>(
		n_output,
		kernel_size,
	    input.data<float>(),
		make_long4(input.size(0), input.size(1), input.size(2), input.size(3)),
		make_long4(input.stride(0), input.stride(1), input.stride(2), input.stride(3)),
	    kernel.data<float>(),
		make_long4(kernel.size(0), kernel.size(1), kernel.size(2), kernel.size(3)),
		make_long4(kernel.stride(0), kernel.stride(1), kernel.stride(2), kernel.stride(3)),
	    output.data<float>(),
		make_long4(output.size(0), output.size(1), output.size(2), output.size(3)),
		make_long4(output.stride(0), output.stride(1), output.stride(2), output.stride(3))
	);

	cudaError_t err = cudaGetLastError();
    // check for errors
    if (err != cudaSuccess) {
    	printf("error in forward_cuda_kernel: %s\n", cudaGetErrorString(err));
    	return 0;
    }

    return 1;
}

*/

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
) {
	int n_output = 0;
	n_output = col*row;
	//int row = tensor_phi.size(0);
	//int col = tensor_phi.size(1);
	//int bx = 16;
	//int by = 16;
	//int gdimX = (col + bx-1) / bx;
    //int gdimY = (row + by-1) / by;
    //dim3 BlockSize(bx, by);
    //dim3 threadsPerBlock(gdimX, gdimY);
	raisr_hash_cuda<<< (n_output + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK, 0, stream >>>(
	    n_output,
	    in_GX.data<float>(),
	    in_GY.data<float>(),
	    hsize,
	    row,
	    col,
	    tensor_phi.data<float>(),
	    tensor_lambda.data<float>(),
	    tensor_mu.data<float>()
	);

	cudaError_t err = cudaGetLastError();
    // check for errors
    if (err != cudaSuccess) {
    	printf("error in forward_cuda_kernel: %s\n", cudaGetErrorString(err));
    	return 0;
    }

    return 1;
}

//*/

#ifdef __cplusplus
	}
#endif
