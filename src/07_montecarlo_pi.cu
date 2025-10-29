#include <iostream>
#include <curand_kernel.h>
#include <cuda_runtime.h>
using namespace std;

__global__ void monteCarloPiKernel(unsigned int *count, unsigned int N, unsigned int seed) {
    
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    unsigned int localCount = 0;

    // random generator
    curandState state;
    curand_init(seed, tid, 0, &state);

    for (unsigned int i = tid; i < N; i += stride){


        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        if (x * x + y * y <= 1.0f){
            localCount++;
        }
           
    }

    atomicAdd(count, localCount);
}

int main() {
    unsigned int N = 1 << 26; // 4m points


    unsigned int *d_count, h_count = 0;
    cudaMalloc(&d_count, sizeof(unsigned int));

    cudaMemcpy(d_count, &h_count, sizeof(unsigned int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = 128;

    monteCarloPiKernel<<<blocks, threads>>>(d_count, N, time(NULL));


    cudaMemcpy(&h_count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    float pi_estimate = 4.0f * h_count / N;



    cout << "Estimated Pi = " << pi_estimate << endl;


    cudaFree(d_count);
    return 0;
}
