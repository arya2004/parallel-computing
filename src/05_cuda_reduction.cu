#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

__global__ void reduceSum(float *input, float *output, int n){


    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    //2 elements per thread
    float sum = 0.0f;

    if(i < n){
        sum += input[i];
    }

    if(i + blockDim.x < n){
        sum += input[i + blockDim.x];
    }

    sdata[tid] = sum;

    __syncthreads();

    //reduce wthin block

    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1){

        if(tid < s) {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();

    }
    //writback
    if(tid == 0){
        output[blockIdx.x] = sdata[0];
    }

}



int main(){

    int n = 1 << 20;
    size_t size = n * sizeof(float);

    vector<float> h_a(n);

    for (int i = 0; i < n; i++){
        h_a[i] = 1.0f;;
       
    }
    

    float *d_a, *d_b;
    cudaMalloc(&d_a, size);

    int threadPerBlock = 256;
    int blocks = (n + threadPerBlock * 2 - 1) / (threadPerBlock * 2);
    size_t outputSize = blocks * sizeof(float);

    cudaMalloc(&d_b, outputSize);


    cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice);
  

    reduceSum<<<blocks, threadPerBlock, threadPerBlock * sizeof(float)>>>(d_a, d_b, n);


    //partial sum back to cpu
    vector<float> h_b(blocks);
    cudaMemcpy(h_b.data(),d_b, outputSize, cudaMemcpyDeviceToHost);


    //final reduction
    float totalSum = 0.0f;

    for (int i = 0; i < blocks; i++){
        totalSum += h_b[i];
    }

    cout << "sum= " << totalSum << endl;
        
    cudaFree(d_a);
    cudaFree(d_b);




    return 0;
}