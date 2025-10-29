#include <cuda_runtime.h>
#include <stdio.h>


__global__ void gpuMul(float *a, float *b, float*c){

    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int k = 512;
    if(r < 256 && col < 256){

        float s = 0.0f;
        for (int i = 0; i < k; i++)
        {
            s += a[r * k + i] * b[i * 256 + col];
        }

        c[r * 256 + col] = s;
        

    }


}

void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

int main(){

    int m = 256;
    int k = 512;
    int n = 256;

    int block_size = 32;


    


}