#include <cuda_runtime.h>
#include <stdio.h>


void cpuAddition(float * a, float *b, float* c, int n){
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
    
}


__global__ void vectorAddition(float * a, float *b, float* c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        c[i] = a[i] + b[i];
    }
}


void initVector(float *vec) {
    for (int i = 0; i < 10000000; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

int main(){

    int n = 10000000;
    int arrSize = 10000000 * sizeof(float);

    float *h_a = (float*) malloc(arrSize);
    float *h_b = (float*) malloc(arrSize);
    float *h_cpu = (float*) malloc(arrSize);
    float *h_c = (float*) malloc(arrSize);

    srand(time(NULL));
    initVector(h_a);
    initVector(h_b);

    float* d_a;
    float* d_b;
    float* d_c;

    cudaMalloc(&d_a, arrSize);
    cudaMalloc(&d_b, arrSize);
    cudaMalloc(&d_c, arrSize);

    cudaMemcpy(d_a, h_a, arrSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, arrSize, cudaMemcpyHostToDevice);

    
    //ceil of /256
    int nb = (n + 256 - 1) / 256;


    cpuAddition(h_a, h_b, h_cpu, n);
    vectorAddition<<<nb, 256>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, arrSize, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (fabs(h_cpu[i] - h_c[i]) > 1e-5) {
            correct = false;
            break;
        }
    }

    printf("Results are %s\n", correct ? "correct" : "incorrect");


    free(h_a);
    free(h_b);
    free(h_cpu);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    printf("dnoe");

    return 0;



}