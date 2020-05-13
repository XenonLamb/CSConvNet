#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define pi 3.141592653589793
using namespace std;
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


    //Q_lamda = len(stre)
    //Q_u = len(cohe)
    //for k in reversed(range(0,len(stre))):
    //    if(lamda < stre[k]):
    //        Q_lamda = k
    //for k in reversed(range(0,len(cohe))):
    //    if(u < cohe[k]):
    //        Q_u = k


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

